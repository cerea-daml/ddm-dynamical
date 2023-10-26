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
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        norm_gamma: torch.Tensor,
        mask: torch.Tensor = None,
        **conditioning
) -> torch.Tensor:
    return (in_data-sigma*prediction) / alpha
