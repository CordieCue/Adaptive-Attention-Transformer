import math
import unittest

import torch

class SinEncoding(torch.nn.Module):
    def __init__(self,hidden_dim,max_len=5000):
        super().__init__()
        pos_emb=torch.zeroes(max_len,hidden_dim)
        