import torch
from torch import nn
import numpy as np


class Propagater(nn.Module):
    def __init__(self, memory, message_dimension, memory_dimension, mean_time_shift_src, neighbor_finder, n_neighbors,
                 tau=2, device='cpu'):
        super(Propagater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device
        self.alpha = 1 / mean_time_shift_src
        self.neighbor_finder = neighbor_finder
        self.n_neighbors = n_neighbors
        self.tau = tau * mean_time_shift_src

        self.tanh = nn.Tanh()
        self.W_s = nn.Parameter(torch.zeros((message_dimension, memory_dimension)).to(self.device))
        self.bias = nn.Parameter(torch.zeros(memory_dimension).to(self.device))


    def compute_time_discount(self, edge_delta):
        time_intervals = self.alpha * edge_delta
        time_discount = torch.exp(-time_intervals)
        return time_discount

    def compute_attention_weight(self, memory, sources_idx, neighbors):
        batch_s = neighbors.shape[0]
        sources = memory[0][sources_idx].view(batch_s, 1, -1)
        softmax = nn.Softmax(dim=1)
        att = softmax(torch.matmul(neighbors, sources.transpose(1, 2)))
        return att

    def forward(self, memory, unique_node_ids, unique_messages, timestamps, inplace=True):
        if len(unique_messages) == 0:
            return memory
        timestamps = timestamps.cpu().numpy()
        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(unique_node_ids, timestamps,
                                                                                      n_neighbors=self.n_neighbors)
        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        batch_s, _ = neighbors.shape
        edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
        edge_deltas = timestamps[:, np.newaxis] - edge_times
        edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
        mask = (torch.from_numpy(edge_deltas).float().to(self.device) > 0).long()
        mask_re = mask.view(batch_s, self.n_neighbors, -1)
        neighbors_cell = memory[0][neighbors_torch.flatten()].view(batch_s, self.n_neighbors, -1)
        edge_delta_re = edge_deltas_torch.view(batch_s, self.n_neighbors, -1)
        unique_messages_re = unique_messages.repeat((self.n_neighbors, 1)).view(batch_s, self.n_neighbors, -1)
        time_discounts = self.compute_time_discount(edge_delta_re)
        time_threshold = (edge_delta_re < self.tau).long()
        att = self.compute_attention_weight(memory, unique_node_ids, neighbors_cell) #(b_s,n_neighbors,1)
        unique_messages = torch.matmul(unique_messages_re, self.W_s) #(b_s,n_neighbors,memory_size)
        C_v = memory[0][neighbors_torch.flatten()] + (mask_re*time_threshold*time_discounts*att * \
              unique_messages).view(batch_s*self.n_neighbors, -1) #(b_s,n_neighbors,memory_size)
        h_v = self.tanh(C_v)
        if inplace:
            memory[0][neighbors_torch.flatten()] = C_v
            memory[1][neighbors_torch.flatten()] = h_v
            return memory
        else:
            memory_cell = memory[0].data
            memory_hidden = memory[1].data
            memory_cell[neighbors_torch.flatten()] = C_v
            memory_hidden[neighbors_torch.flatten()] = h_v
            return [memory_cell, memory_hidden]


