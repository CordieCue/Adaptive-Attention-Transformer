import unittest
import math
import typing
from typing import Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from attention import Attention
from positional_encoding import SinEncoding
from vocabulary import Vocabulary

class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 ff_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout_p: float):
        super().__init__()
        self.embed = embedding
        self.positional_encoding=SinEncoding(hidden_dim,max_len=5000)
        self.hidden_dim=hidden_dim
        self.Dropout=nn.Dropout(dropout_p)
        self.encoder_blocks=nn.ModuleList(
            [
                EncoderBlock(hidden_dim,ff_dim,num_heads,dropout_p) for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    def forward(self,x:torch.tensor,mask:Optional[torch.Tensor]=None)->torch.Tensor:
        x=self.embed(x)*math.sqrt(self.hidden_dim)
        x=self.positional_encoding(x)
        x=self.Dropout(x)
        for encoder_block in self.encoder_blocks:
            x=encoder_block.forward(x,mask=mask)
        return x

class EncoderBlock(nn.Module):
    def __init__(self,hidden_dim:int,ff_dim:int,num_heads:int,dropout_p:float):
        super().__init__()
        self.attention=Attention(hidden_dim,num_heads,dropout_p)
        self.norm1=nn.LayerNorm(hidden_dim)
        self.ff=nn.Sequential(
            nn.Linear(hidden_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,hidden_dim)
        )
        self.norm2=nn.LayerNorm(hidden_dim)
        self.dropout=nn.Dropout(dropout_p)

    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor]=None)->torch.Tensor:
        attn_output=self.attention.forward(x,x,x,mask=mask)
        x=self.norm1(x+self.dropout(attn_output))
        ff_output=self.ff(x)
        x=self.norm2(x+self.dropout(ff_output))
        return x