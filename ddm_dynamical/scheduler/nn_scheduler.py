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

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return F.linear(in_tensor, self.weight.abs(), self.bias)


class NNScheduler(NoiseScheduler):
    def __init__(
            self,
            n_features: int = 1024,
            gamma_min: float = -10,
            gamma_max: float = 10,
    ):
        super().__init__(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        self.n_features = n_features
        self.l1 = LinearMonotonic(1, 1)
        torch.nn.init.constant_(self.l1.weight, gamma_max-gamma_min)
        torch.nn.init.constant_(self.l1.bias, gamma_min)
        self.l2 = LinearMonotonic(1, self.n_features)
        torch.nn.init.normal_(self.l2.weight)
        self.activation = torch.nn.Sigmoid()
        self.l3 = LinearMonotonic(self.n_features, 1, bias=False)
        torch.nn.init.normal_(self.l3.weight)
        self.branch_factor = torch.nn.Parameter(torch.ones(1)*1E-6)

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        time_tensor = 1.-timesteps[..., None]
        output = self.l1(time_tensor)
        branch = (time_tensor-0.5)*2.
        branch = self.l2(branch)
        branch = self.activation(branch)
        branch = 2*(branch-0.5)
        output += self.l3(branch) / self.n_features * self.branch_factor
        return output.squeeze(dim=-1)
