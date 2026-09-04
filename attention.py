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

        w_q,w_kv=self.qkv_proj.weight.split(hidden_dim,2*hidden_dim)

        k,v=F.linear(encoder_hidden_states,w_kv).reshape(batch_size,src_seq_length,self.num_heads,self.head_dim).chunk(2,dim=-1)
        q=F.linear(decoder_hidden_states,w_q).reshape(batch_size,tgt_seq_length,self.num_heads,self.head_dim)
        return q,k,v

    def scaled_dot_product_attention(self,q:torch.tensor,
                                     k:torch.tensor,
                                     v:torch.tensor,
                                     attention_mask:Optional[torch.tensor]=None,
                                     future_mask:Optional[torch.tensor]=None):
        batch_size,num_heads,seq_length,head_dim=q.size()
        logits=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(head_dim)
        if attention_mask is not None:
            logits=self.mask_logits(logits,attention_mask,future_mask)
        attention=F.softmax(logits,dim=-1)
        values=torch.matmul(attention,v)
        return values,attention
    
    @staticmethod
    def mask_logits(self,logits:torch.tensor,
                    attention_mask:Optional[torch.tensor]=None,
                    future_mask:Optional[torch.tensor]=None):
        masked_logits=logits
        if attention_mask is not None:
            masked_logits=logits.masked_fill(attention_mask==0,float('-inf'))
        if future_mask is not None:
            masked_logits=logits.masked_fill(future_mask==0,float('-inf'))
        return masked_logits
        