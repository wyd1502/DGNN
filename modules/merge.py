import torch
from torch import nn

from collections import defaultdict


class MemoryMerge(nn.Module):
    def __init__(self, memory_dimension, device='cpu'):
        super(MemoryMerge, self).__init__()
        self.device = device
        self.W1 = nn.Parameter(torch.zeros((memory_dimension, memory_dimension)).to(self.device))
        self.W2 = nn.Parameter(torch.zeros((memory_dimension, memory_dimension)).to(self.device))
        self.bias = nn.Parameter(torch.zeros(memory_dimension).to(self.device))
        self.act = torch.nn.ReLU()

    def forward(self, memory_s, memory_g):
        return torch.matmul(memory_s, self.W1)+torch.matmul(memory_g, self.W2) + self.bias
