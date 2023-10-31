#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 23/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable

# External modules
import torch

# Internal modules
from .binarized_scheduler import BinarizedScheduler


main_logger = logging.getLogger(__name__)


class PredefinedScheduler(BinarizedScheduler):
    def __init__(
            self,
            bin_values: Iterable[float],
            gamma_min: float = -15,
            gamma_max: float = 10,
    ):
        """
        Binarized noise scheduler as proposed in
        Kingma and Guo, `Understanding Diffusion Objectives as the ELBO with
        Simple Data Augmentation`.
        """
        super().__init__(
            n_bins=len(bin_values), gamma_min=gamma_min, gamma_max=gamma_max
        )
        self.bin_values = torch.tensor(bin_values)
        self._update_times()

    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        pass

