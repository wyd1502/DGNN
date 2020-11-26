from torch import nn
import torch


class MessageFunction(nn.Module):
  def __init__(self, memory_dimension, message_dimension, edge_dimension=0, device="cpu"):
    super(MessageFunction, self).__init__()
    self.device = device
    self.W1 = nn.Parameter(torch.zeros((memory_dimension, message_dimension,)).to(self.device))
    self.W2 = nn.Parameter(torch.zeros((memory_dimension, message_dimension)).to(self.device))
    self.W3 = nn.Parameter(torch.zeros((edge_dimension, message_dimension)).to(self.device))
    nn.init.xavier_uniform_(self.W1)
    nn.init.xavier_uniform_(self.W2)
    self.bias = nn.Parameter(torch.zeros(message_dimension).to(self.device))
    self.act = nn.ReLU()

  def compute_message(self, memory_s, memory_g, edge_fea=None):
    messages = self.act(torch.matmul(memory_s, self.W1)+torch.matmul(memory_g, self.W2,) +
                        torch.matmul(edge_fea, self.W3)+self.bias)
    #messages = self.bn1(messages)
    return messages


