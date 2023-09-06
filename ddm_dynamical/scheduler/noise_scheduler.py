#!/bin/env python
# -*- coding: utf-8 -*-
#
#
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import logging
import abc

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class NoiseScheduler(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_gamma(self, timestep: torch.Tensor) -> torch.Tensor:
        pass

    def get_snr(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.get_gamma(timestep))

    def get_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-self.get_gamma(timestep))

    def get_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.get_gamma(timestep))
