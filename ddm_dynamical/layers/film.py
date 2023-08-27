#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 13/12/2022
# Created for 2022_ddim_for_attractors
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


class FilmLayer(torch.nn.Module):
    def __init__(
            self,
            n_neurons: int,
            n_conditional: int,
    ):
        super().__init__()
        self.affine_film = torch.nn.Linear(
            n_conditional, n_neurons*2
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedded_time: torch.Tensor
    ) -> torch.Tensor:
        scale_shift = self.affine_film(embedded_time)
        scale_shift = scale_shift.view(
            scale_shift.shape+(1, )*(in_tensor.dim()-scale_shift.dim())
        )
        scale, shift = scale_shift.chunk(2, dim=1)
        filmed_tensor = in_tensor * (scale + 1) + shift
        return filmed_tensor
