import math
import torch 
from typing import Optional
import unittest
from torch import nn
from torch.nn import functional as F
from utils import construct_fututre_mask

class attentiom(nn.module):
    def __init__(self.hidden_dim:int,num_heads:int):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.head_dim=hidden_dim//num_heads
        self.qkv_proj=nn.Linear(hidden_dim,3*hidden_dim)
        self.o_proj=nn.Linear(hidden_dim,hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.constant_(self.qkv_proj.bias,0.)
        nn.init.constant_(self.o_proj.bias,0.)

    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor]=None)->torch.Tensor:
        batch_size,seq_len,_=x.size()
        qkv=self.qkv_proj(x)
        qkv=qkv.reshape(batch_size,seq_len,self.num_heads,3*self.head_dim)
        qkv=qkv.permute(0,2,1,3)
        q,k,v=qkv.chunk(3,dim=-1)
        attn_scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores=attn_scores.masked_fill(mask==0,float('-inf'))
        attn_weights=F.softmax(attn_scores,dim=-1)
        attn_output=torch.matmul(attn_weights,v)
        attn_output=attn_output.permute(0,2,1,3).reshape(batch_size,seq_len,self.hidden_dim)
        output=self.o_proj(attn_output)
        return output

    def test_attention(self):
        batch_size=2
        seq_len=4
        hidden_dim=8
        num_heads=2
        x=torch.rand(batch_size,seq_len,hidden_dim)
        mask=construct_future_mask(seq_len)
        attention_layer=attention(hidden_dim,num_heads)
        output=attention_layer(x,mask)
        self.assertEqual(output.shape,(batch_size,seq_len,hidden_dim))                      

    