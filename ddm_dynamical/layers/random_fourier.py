#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/12/2022
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
import math


# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class RandomFourierLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int = 3,
            n_neurons: int = 512,
            wave_length: float = 1.
    ) -> None:
        super().__init__()
        self.register_parameter(
            "weights",
            torch.nn.Parameter(
                torch.randn(in_features, n_neurons//2), requires_grad=False
            )
        )
        self.constant = math.sqrt(2 / n_neurons)
        self.wave_length = wave_length

    def extract_features(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return self(in_tensor)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_tensor = in_tensor / self.wave_length
        out_tensor = 2 * torch.pi * in_tensor @ self.weights
        out_tensor = self.constant * torch.cat(
            [torch.sin(out_tensor), torch.cos(out_tensor)], dim=-1
        )
        return out_tensor
