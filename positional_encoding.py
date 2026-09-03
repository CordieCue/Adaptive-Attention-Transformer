import math
import unittest

import torch

class SinEncoding(torch.nn.Module):
    def __init__(self,hidden_dim,max_len=5000):
        super().__init__()
        pos_emb=torch.zeroes(max_len,hidden_dim)
        pos=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,hidden_dim,2)*(-math.log(10000.0)/hidden_dim))
        pos_emb[:,0::2]=torch.sin(pos*div_term)
        pos_emb[:,1::2]=torch.cos(pos*div_term)
        pos_emb=pos_emb.unsqueeze(0).transpose(0,1)
        self.register_buffer('pos_emb',pos_emb)

    def forward(self,x):
        x=x+self.pos_emb[:x.size(0),:]
        return x