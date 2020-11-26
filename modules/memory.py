import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, message_dimension=None,
               device="cpu"):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.message_dimension = message_dimension
    self.device = device


    self.__init_memory__()

  def __init_memory__(self, seed=0):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    torch.manual_seed(seed)
    self.cell = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.hidden = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.memory = [self.cell, self.hidden]
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)
    nn.init.xavier_normal(self.hidden)
    nn.init.xavier_normal(self.cell)
    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return [self.memory[i][node_idxs, :] for i in range(2)]

  def set_cell(self, node_idxs, values):
    self.cell[node_idxs, :] = values

  def set_hidden(self, node_idxs, values):
    self.hidden[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return [self.memory[i].data.clone() for i in range(2)], self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.cell.data, self.hidden.data, self.last_update.data = \
      memory_backup[0][0].clone(), memory_backup[0][1].clone(), memory_backup[1].clone()
    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    self.hidden.detach_()
    self.cell.detach_()

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
