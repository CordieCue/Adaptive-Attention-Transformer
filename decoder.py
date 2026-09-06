import math
import unittest
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from attention import Attention
from positional_encoding import SinEncoding
from utils import construct_future_mask

class TransformerDecoder(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 ff_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 vocab_size: int,
                 dropout_p: float,
                 tie_output_embedding: bool = True):
        super().__init__()
        self.embed = embedding
        self.positional_encoding=SinEncoding(hidden_dim,max_len=5000)
        self.hidden_dim=hidden_dim
        self.Dropout=nn.Dropout(dropout_p)
        self.decoder_blocks=nn.ModuleList(
            [
                DecoderBlock(hidden_dim,ff_dim,num_heads,dropout_p) for _ in range(num_layers)
            ]
        )
        self.output_layer=nn.Linear(hidden_dim,vocab_size)
        if tie_output_embedding:
            self.output_layer.weight=self.embed.weight


    def _reset_parameters(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    def forward(self,x:torch.tensor,
                input_tokens:Optional[torch.Tensor]=None,
                encoder_hidden_states:Optional[torch.Tensor]=None,
                src_mask:Optional[torch.Tensor]=None,
                future_mask:Optional[torch.Tensor]=None)->torch.Tensor:

        x=self.embed(x)*math.sqrt(self.hidden_dim)
        x=self.positional_encoding(x)
        x=self.Dropout(x)
        for decoder_block in self.decoder_blocks:
            x=decoder_block.forward(x,encoder_hidden_states,src_mask,future_mask)
        logits=self.output_layer(x)
        return logits

class DecoderBlock(nn.Module):
    def __init__(self,hidden_dim:int,ff_dim:int,num_heads:int,dropout_p:float):
        super().__init__()
        self.cross_mha = Attention(hidden_dim, num_heads)
        self.self_mha = Attention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self,
                x:torch.Tensor,
                input_tokens:torch.Tensor,
                encoder_hidden_states:torch.Tensor,
                src_mask:Optional[torch.Tensor]=None,
                future_mask:Optional[torch.Tensor]=None)->torch.Tensor:

        output=self.dropout1(self.self_mha.forward(x,future_mask=future_mask))
        x=self.layer_norm1(x+output)

        output=self.dropout2(self.cross_mha.forward(x,encoder_hidden_states,src_mask=src_mask))
        x=self.layer_norm2(x+output)

        output=self.dropout3(self.feed_forward(x))
        x=self.layer_norm3(x+output)

        return x
    
        