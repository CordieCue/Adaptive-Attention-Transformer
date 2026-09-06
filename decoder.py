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
                 dropout_p: float):
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

    def _reset_parameters(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    def forward(self,x:torch.tensor,encoder_hidden_states:Optional[torch.Tensor]=None,mask:Optional[torch.Tensor]=None)->torch.Tensor:
        x=self.embed(x)*math.sqrt(self.hidden_dim)
        x=self.positional_encoding(x)
        x=self.Dropout(x)
        for decoder_block in self.decoder_blocks:
            x=decoder_block.forward(x,encoder_hidden_states,mask)
        return x