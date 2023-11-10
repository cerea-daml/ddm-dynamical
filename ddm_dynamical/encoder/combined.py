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
from typing import List

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class CombinedEncoder(torch.nn.Module):
    def __init__(self, base_encoder: List[torch.nn.Module]):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                encoder(in_tensor[:, [k]])
                for k, encoder in enumerate(self.base_encoders)
            ], dim=1
        )
