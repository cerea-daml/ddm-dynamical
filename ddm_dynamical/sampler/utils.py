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
        prediction: torch.Tensor,
        in_data: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
        time_tensor: torch.Tensor,
        mask: torch.Tensor = None,
        **conditioning
) -> torch.Tensor:
    return (in_data-sigma_t*prediction) / alpha_t
