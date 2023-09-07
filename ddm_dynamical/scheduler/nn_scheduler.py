#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/09/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch.nn
import torch.nn.functional as F

# Internal modules
from .noise_scheduler import NoiseScheduler

main_logger = logging.getLogger(__name__)


class LinearMonotonic(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.proj = torch.nn.Softplus()

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return F.linear(in_tensor, self.proj(self.weight), self.bias)


class NNScheduler(NoiseScheduler):
    def __init__(
            self,
            n_features=1024,
            min_gamma: float = -5,
            max_gamma: float = 5,
            normalize: bool = True
    ):
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.normalize = normalize
        self.n_features = n_features
        self.l1 = LinearMonotonic(1, 1)
        torch.nn.init.constant_(self.l1.weight, max_gamma-min_gamma)
        torch.nn.init.constant_(self.l1.bias, min_gamma)
        self.l2 = LinearMonotonic(1, self.n_features)
        torch.nn.init.normal_(self.l2.weight)
        self.activation = torch.nn.Sigmoid()
        self.l3 = LinearMonotonic(self.n_features, 1, bias=False)

    def forward(self, time_tensor: torch.Tensor) -> torch.Tensor:
        time_tensor = time_tensor[..., None]
        output = self.l1(time_tensor)
        branch = (time_tensor-0.5)*2.
        branch = self.l2(branch)
        branch = self.activation(branch)
        branch = 2*(branch-0.5)
        output += self.l3(branch) / self.n_features
        return output.squeeze(dim=-1)

    def normalize_gamma(self, gamma:  torch.Tensor) -> torch.Tensor:
        gamma_ends = self(
            torch.tensor([0., 1.], dtype=gamma.dtype, device=gamma.device)
        )
        scale = (self.max_gamma-self.min_gamma)/(gamma_ends[1]-gamma_ends[0])
        return self.min_gamma + scale * (gamma-gamma_ends[0])

    def get_gamma(self, time_tensor: torch.Tensor) -> torch.Tensor:
        gamma = self(time_tensor)
        if self.normalize:
            gamma = self.normalize_gamma(gamma)
        return gamma
