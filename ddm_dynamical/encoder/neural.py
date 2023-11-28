#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 28/11/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging


# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class NeuralEncoder(torch.nn.Module):
    def __init__(self, network: torch.nn.Module, physical_encoder: torch.nn.Module = None):
        super().__init__()
        self.network = network
        if physical_encoder is None:
            self.physical_encoder = lambda x, mask: x
        else:
            self.physical_encoder = physical_encoder

    def forward(
            self,
            in_tensor: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.physical_encoder(in_tensor, mask)
        return self.network(encoded, mask)
