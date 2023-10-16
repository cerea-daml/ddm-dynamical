#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 16/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


def project_to_state(
        in_data: torch.Tensor,
        prediction: torch.Tensor,
        alpha_t: torch.Tensor = 1.,
        sigma_t: torch.Tensor = 1.
) -> torch.Tensor:
    return (in_data-sigma_t*prediction) / alpha_t
