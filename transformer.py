import random
import unittest
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from vocabulary import Vocabulary
from attention import Attention
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import construct_future_mask

class Transformer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 ff_dim: int,
                 num_heads: int,
                 num_layers: int,
                 max_decoding_len: int,
                 vocab_size: int,
                 padding_idx: int,
                 bos_idx: int,
                 dropout_p: float,
                 tie_output_embedding: Optional[bool] = True):

        super().__init__()
        self.embed=nn.Embedding(vocab_size,hidden_dim,padding_idx=padding_idx)

        self.encoder=TransformerEncoder(self.embed,
                                        ff_dim,hidden_dim,
                                        num_heads,
                                        num_layers,
                                        dropout_p)
        
        self.decoder=TransformerDecoder(self.embed,
                                        ff_dim,hidden_dim,
                                        num_heads,num_layers,
                                        vocab_size,dropout_p,
                                        max_decoding_len,
                                        tie_output_embedding)

        self.bos_idx=bos_idx
        self.padding_idx=padding_idx
        self.vocab_size=vocab_size
        self.max_decoding_len=max_decoding_len
        self.hidden_dim=hidden_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class TestTransformer(unittest.TestCase):
    def test_transformer_inference(self):
        seed=0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        corpus=["hello world","this is a test","transformers are great",
                "i love natural language processing","deep learning is fun"]

        en_vocab=Vocabulary(corpus)
        en_vocab_size=len(en_vocab.token2index.items())
        


