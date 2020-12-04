import torch
from torch import nn

from collections import defaultdict


class MemoryUpdater(nn.Module):
  def __init__(self, memory, message_dimension, memory_dimension, mean_time_shift_src, device):
    super(MemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device
    self.alpha = 2

    self.sig = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.W_d = nn.Parameter(torch.zeros((memory_dimension,memory_dimension)).to(self.device))
    self.b_d = nn.Parameter(torch.zeros(memory_dimension).to(self.device))
    self.W_f = nn.Parameter(torch.zeros((memory_dimension, message_dimension)).to(self.device))
    self.U_f = nn.Parameter(torch.zeros((memory_dimension, memory_dimension)).to(self.device))
    self.b_f = nn.Parameter(torch.zeros(memory_dimension).to(self.device))
    self.W_i = nn.Parameter(torch.zeros((memory_dimension, message_dimension)).to(self.device))
    self.U_i = nn.Parameter(torch.zeros((memory_dimension, memory_dimension)).to(self.device))
    self.b_i = nn.Parameter(torch.zeros(memory_dimension).to(self.device))
    self.W_o = nn.Parameter(torch.zeros((memory_dimension, message_dimension)).to(self.device))
    self.U_o = nn.Parameter(torch.zeros((memory_dimension, memory_dimension)).to(self.device))
    self.b_o = nn.Parameter(torch.zeros(memory_dimension).to(self.device))
    self.W_c = nn.Parameter(torch.zeros((memory_dimension, message_dimension)).to(self.device))
    self.U_c = nn.Parameter(torch.zeros((memory_dimension, memory_dimension)).to(self.device))
    self.b_c = nn.Parameter(torch.zeros(memory_dimension).to(self.device))
  def update_memory(self, unique_node_ids, unique_messages, timestamps, inplace=True):
    if len(unique_messages) == 0:
      return self.memory.memory, self.memory.last_update
    hidden = self.memory.memory[1][unique_node_ids] #hidden [bs,memroy_size]
    cell = self.memory.memory[0][unique_node_ids]  #cell [bs,memroy_size]
    messages = unique_messages
    time_discounts = self.compute_time_discount(unique_node_ids, timestamps) #[bs,1]
    bs = hidden.shape[0]
    C_vi = self.tanh(torch.matmul(self.W_d, cell.t()).t()+self.b_d) # [bs,memory_size]
    C_v_discount = torch.mul(C_vi, time_discounts.view(bs, 1))
    C_v_t = cell - C_v_discount
    C_v_star = C_v_t + C_v_discount
    f_t = self.sig(torch.matmul(self.W_f, messages.t()).t() + torch.matmul(self.U_f, hidden.t()).t()+self.b_f)
    i_t = self.sig(torch.matmul(self.W_i, messages.t()).t() + torch.matmul(self.U_i, hidden.t()).t() + self.b_i)
    o_t = self.sig(torch.matmul(self.W_o, messages.t()).t() + torch.matmul(self.U_o, hidden.t()).t() + self.b_o)
    C_hat_t = self.tanh(torch.matmul(self.W_c, messages.t()).t() + torch.matmul(self.U_c, hidden.t()).t() + self.b_c)
    C_v_t = torch.mul(f_t, C_v_star) + torch.mul(i_t, C_hat_t)
    h_v_t = torch.mul(o_t, self.tanh(C_v_t))
    if inplace:
      self.memory.memory[0][unique_node_ids] = C_v_t
      self.memory.memory[1][unique_node_ids] = h_v_t
      self.memory.last_update[unique_node_ids] = timestamps
      return self.memory.memory, self.memory.last_update
    else:
      memory_cell = self.memory.memory[0].data.clone()
      memory_hidden = self.memory.memory[1].data.clone()
      last_update = self.memory.last_update.data.clone()
      memory_cell[unique_node_ids] = C_v_t
      memory_hidden[unique_node_ids] = h_v_t
      last_update[unique_node_ids] = timestamps
      return [memory_cell, memory_hidden], last_update


  def compute_time_discount(self,unique_node_ids, timestamps):
    time_intervals = timestamps - self.memory.last_update[unique_node_ids]
    time_intervals = self.alpha*time_intervals
    time_discount = torch.exp(-time_intervals)
    return time_discount
