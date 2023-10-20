#!/bin/env python
# -*- coding: utf-8 -*-
#
#
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import logging
import math

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim: int = 512, max_freq: float = 10000):
        super().__init__()
        half_dim = dim // 2
        embeddings = math.log(max_freq) / (half_dim - 1)
        self.register_buffer(
            "frequencies", torch.exp(torch.arange(half_dim,) * -embeddings)
        )

    def forward(
            self, in_tensor: torch.Tensor
    ) -> torch.Tensor:
        embedding = in_tensor * self.frequencies
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding
