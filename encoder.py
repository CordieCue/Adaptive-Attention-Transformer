import unittest
import math
import typing
from typing import Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from attention import Attention
from positional_encoding import PositionalEncoding
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
