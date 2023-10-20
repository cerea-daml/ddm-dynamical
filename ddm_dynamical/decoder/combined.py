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


class CombinedDecoder(torch.nn.Module):
    def __init__(self, base_decoders: Iterable[torch.nn.Module]):
        super().__init__()
        self.base_decoders = base_decoders

    def forward(
            self, in_tensor: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if in_tensor.size(1) != len(self.base_decoders):
            raise ValueError(
                "The number of channels and base decoder have to be the same!"
            )
        return torch.cat(
            [
                decoder(in_tensor[:, k])
                for k, decoder in enumerate(self.base_decoders)
            ], dim=1
        )


