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
from typing import Dict

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class CombinedEncoder(torch.nn.Module):
    def __init__(self, base_encoders: Dict[str, torch.nn.Module]):
        super().__init__()
        self.base_encoders = torch.nn.ModuleDict(base_encoders)

    def forward(
            self,
            in_tensor: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [
                encoder(in_tensor[:, [k]], mask)
                for k, encoder in enumerate(self.base_encoders.values())
            ], dim=1
        )
