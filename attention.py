import math
import torch 
from typing import Optional
import unittest
from torch import nn
from torch.nn import functional as F
from utils import construct_future_mask


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim,bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)


    def forward(self,
                x:torch.tensor,
                encoder_hidden_states:Optional[torch.tensor]=None,
                attention_mask:Optional[torch.tensor]=None,
                future_mask:Optional[torch.tensor]=None):
            
        batch_size, seq_length, hidden_dim=x.size()
        if encoder_hidden_states is  None:
            q,k,v=self.self_attention(x)
        else:
            q,k,v=self.cross_attention(x,encoder_hidden_states)
        values,attention=self.scaled_dot_product_attention(q,k,v,attention_mask,future_mask)

        q=q.permute(0,2,1,3)
        k=k.permute(0,2,1,3)   
        v=v.permute(0,2,1,3)

        values,attention=self.scaled_dot_product_attention(q,k,v,attention_mask,future_mask)
        values=values.permute(0,2,1,3).reshape(batch_size,seq_length,self.hidden_dim)

        return self.o_proj(values)

    def self_attention(self,x:torch.tensor):
        batch_size,seq_length,hidden_dim=x.size()
        qkv=self.qkv_proj(x)
        q,k,v=qkv.chunk(3,dim=-1)
        return q,k,v

    def cross_attention(self,x:torch.tensor,
                        encoder_hidden_states:torch.tensor,
                        decoder_hidden_states:torch.tensor):
        
        batch_size,src_seq_length,hidden_dim=encoder_hidden_states.size()
        batch_size,tgt_seq_length,hidden_dim=decoder_hidden_states.size()

        
        