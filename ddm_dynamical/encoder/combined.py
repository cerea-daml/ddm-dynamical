#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class CombinedEncoder(torch.nn.Module):
    def __init__(self, base_encoders: Iterable[torch.nn.Module]):
        super().__init__()
        self.base_encoders = base_encoders

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        if in_tensor.size(1) != len(self.base_encoders):
            raise ValueError(
                "The number of channels and base encoders have to be the same!"
            )
        return torch.cat(
            [
                encoder(in_tensor[:, k])
                for k, encoder in enumerate(self.base_encoders)
            ], dim=1
        )
