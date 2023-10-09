#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 09/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Any

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


def masked_average(
        in_tensor: torch.Tensor,
        mask: torch.Tensor = None,
        dim: Any = None,
        eps: float = 1E-9
) -> torch.Tensor:
    in_masked_sum = torch.sum(in_tensor*mask, dim=dim)
    expanded_mask = torch.ones_like(in_tensor)
    if mask is not None:
        expanded_mask *= mask
    mask_sum = torch.sum(expanded_mask, dim=dim)
    return in_masked_sum / (mask_sum + eps)
