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


def sample_time(
        template_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Samples time indices as tensor between [0, 1]. The time indices are
    equidistant distributed to reduce the training variance as proposed in
    Kingma et al., 2021.

    Parameters
    ----------
    template_tensor : torch.Tensor
        The template tensor with `n` dimensions and shape (batch size, *).

    Returns
    -------
    sampled_time : torch.Tensor
        The time tensor sampled for each sample independently. The resulting
        shape is (batch size, *) with `n` dimensions filled by ones. The
        tensor lies on the same device as the input.
    """
    time_shape = torch.Size(
        [template_tensor.size(0)] + [1, ] * (template_tensor.ndim - 1)
    )
    # Draw initial time
    time_shift = torch.randn(
        1, dtype=template_tensor.dtype, device=template_tensor.device
    )
    # Equidistant timing
    sampled_time = torch.linspace(
        0, 1, template_tensor.size(0) + 1,
        dtype=template_tensor.dtype, device=template_tensor.device
    )[:template_tensor.size(0)]
    # Estimate time
    sampled_time = (time_shift + sampled_time) % 1
    sampled_time = sampled_time.reshape(time_shape)
    return sampled_time


def normalize_gamma(
        gamma: Any, gamma_min: float = -15., gamma_max: float = 15
) -> Any:
    return (gamma-gamma_min) / (gamma_max-gamma_min)


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
