import torch
from torch import nn
from collections import defaultdict
import logging
import numpy as np

from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.merge import MemoryMerge
from modules.message_function import MessageFunction
from modules.update import MemoryUpdater
from modules.propagater import Propagater


class DGNN(nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device,
                 dropout=0.1,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=200, n_neighbors=None, aggregator_type="last",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1):
        super(DGNN, self).__init__()
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.memory_s = None
        self.memory_g = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst
        self.memory_dimension = memory_dimension
        self.memory_update_at_start = memory_update_at_start
        self.message_dimension = message_dimension
        self.memory_merge = MemoryMerge(self.memory_dimension, self.device)
        self.memory_s = Memory(n_nodes=self.n_nodes,
                               memory_dimension=self.memory_dimension,
                               message_dimension=message_dimension,
                               device=device)
        self.memory_g = Memory(n_nodes=self.n_nodes,
                               memory_dimension=self.memory_dimension,
                               message_dimension=message_dimension,
                               device=device)
        self.message_dim = message_dimension
        self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                         device=device)
        self.message_function = MessageFunction(memory_dimension=memory_dimension,
                                                message_dimension=message_dimension, edge_dimension=self.n_edge_features,
                                                device=self.device)
        self.memory_updater_s = MemoryUpdater(memory=self.memory_s, message_dimension=message_dimension,
                                              memory_dimension=self.memory_dimension,
                                              mean_time_shift_src=self.mean_time_shift_src/2,
                                              device=self.device)
        self.memory_updater_g = MemoryUpdater(memory=self.memory_g, message_dimension=message_dimension,
                                              memory_dimension=self.memory_dimension,
                                              mean_time_shift_src=self.mean_time_shift_dst / 2,
                                              device=self.device)
        self.propagater_s = Propagater(memory=self.memory_s, message_dimension=message_dimension,
                                       memory_dimension=self.memory_dimension,
                                       mean_time_shift_src=self.mean_time_shift_src / 2,
                                       neighbor_finder=self.neighbor_finder, n_neighbors=self.n_neighbors, tau=2,
                                       device=self.device)
        self.propagater_g = Propagater(memory=self.memory_g, message_dimension=message_dimension,
                                       memory_dimension=self.memory_dimension,
                                       mean_time_shift_src=self.mean_time_shift_dst / 2,
                                       neighbor_finder=self.neighbor_finder, n_neighbors=self.n_neighbors, tau=2,
                                       device=self.device)
        self.W_s = nn.Parameter(torch.zeros((memory_dimension, memory_dimension // 2)).to(self.device))
        #nn.xavier_
        self.W_g = nn.Parameter(torch.zeros((memory_dimension, memory_dimension // 2)).to(self.device))

    def update_memory(self, source_nodes, destination_nodes, messages_s, messages_g):
        # Aggregate messages for the same nodes

        unique_src_nodes, unique_src_messages, unique_src_timestamps = self.message_aggregator.aggregate(source_nodes,
                                                                                                         messages_s)
        unique_des_nodes, unique_des_messages, unique_des_timestamps = self.message_aggregator.aggregate(destination_nodes,
                                                                                                         messages_g)

        # Update the memory with the aggregated messages
        self.memory_updater_s.update_memory(unique_src_nodes, unique_src_messages,
                                            timestamps=unique_src_timestamps)
        self.memory_updater_g.update_memory(unique_des_nodes, unique_des_messages,
                                            timestamps=unique_des_timestamps)

    def propagate(self, source_nodes, destination_nodes, messages_s, messages_g):
        unique_src_nodes, unique_src_messages, unique_src_timestamps = self.message_aggregator.aggregate(source_nodes,
                                                                                                         messages_s)
        unique_des_nodes, unique_des_messages, unique_des_timestamps = self.message_aggregator.aggregate(
            destination_nodes,
            messages_g)

        self.propagater_s(self.memory_s.memory, unique_src_nodes, unique_src_messages,
                          timestamps=unique_src_timestamps)
        self.propagater_g(self.memory_g.memory, unique_des_nodes, unique_des_messages,
                          timestamps=unique_des_timestamps)

    def compute_loss(self, memory_s, memory_g, source_nodes, destination_nodes):
        source_mem = self.memory_merge(memory_s[1][source_nodes], memory_g[1][source_nodes])
        destination_mem = self.memory_merge(memory_s[1][destination_nodes], memory_g[1][destination_nodes])
        source_emb = torch.matmul(source_mem, self.W_s)
        destination_emb = torch.matmul(destination_mem, self.W_g)
        score = torch.sum(source_emb*destination_emb, dim=1)
        return score.sigmoid()

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, test=False):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])
        memory = None
        time_diffs = None
        memory_s, last_update_s, memory_g, last_update_g = \
            self.get_updated_memory(list(range(self.n_nodes)), list(range(self.n_nodes)),
                                    self.memory_s.messages, self.memory_g.messages)

        pos_score = self.compute_loss(memory_s, memory_g, source_nodes, destination_nodes)
        neg_score = self.compute_loss(memory_s, memory_g, source_nodes, negative_nodes)
        self.update_memory(source_nodes, destination_nodes,
                               self.memory_s.messages, self.memory_g.messages)
        self.propagate(source_nodes, destination_nodes,
                           self.memory_s.messages, self.memory_g.messages)
        self.memory_s.clear_messages(positives)
        self.memory_g.clear_messages(positives)
        unique_sources, source_id_to_messages = self.get_messages(source_nodes,
                                                                  destination_nodes,
                                                                  edge_times, edge_idxs)
        unique_destinations, destination_id_to_messages = self.get_messages(destination_nodes,
                                                                            source_nodes,
                                                                            edge_times,
                                                                            edge_idxs)
        self.memory_s.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory_g.store_raw_messages(unique_destinations, destination_id_to_messages)
        if not test:
            return pos_score, neg_score
        else:
            source_mem = self.memory_merge(memory_s[1][source_nodes], memory_g[1][source_nodes])
            destination_mem = self.memory_merge(memory_s[1][destination_nodes], memory_g[1][destination_nodes])
            return source_mem, destination_mem

    def get_messages(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]
        source_memory = self.memory_merge(self.memory_s.memory[1][source_nodes],
                                          self.memory_g.memory[1][source_nodes])
        destination_memory = self.memory_merge(self.memory_s.memory[1][destination_nodes],
                                               self.memory_g.memory[1][destination_nodes])

        source_message = self.message_function.compute_message(source_memory, destination_memory, edge_features)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def get_updated_memory(self, source_nodes, destination_nodes, message_s, message_g):
        unique_src_nodes, unique_src_messages, unique_src_timestamps = self.message_aggregator.aggregate(source_nodes,
                                                                                                         message_s)
        unique_des_nodes, unique_des_messages, unique_des_timestamps = self.message_aggregator.aggregate(destination_nodes,
                                                                                                         message_g)
        updated_src_memory, updated_src_last_update = self.memory_updater_s.update_memory(unique_src_nodes,
                                                                                          unique_src_messages,
                                                                                   timestamps=unique_src_timestamps,
                                                                                   inplace=False)
        updated_des_memory, updated_des_last_update = self.memory_updater_g.update_memory(unique_des_nodes,
                                                                                          unique_des_messages,
                                                                                  timestamps=unique_des_timestamps,
                                                                                  inplace=False)
        updated_src_memory = self.propagater_s(updated_src_memory, unique_src_nodes, unique_src_messages,
                                               timestamps=unique_src_timestamps, inplace=False)
        updated_des_memory = self.propagater_g(updated_des_memory, unique_des_nodes, unique_des_messages,
                                               timestamps=unique_des_timestamps, inplace=False)

        return updated_src_memory, updated_src_last_update, updated_des_memory, updated_des_last_update

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.propagater_s.neighbor_finder = neighbor_finder
        self.propagater_g.neighbor_finder = neighbor_finder

