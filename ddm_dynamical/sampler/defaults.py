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


__all__ = [
    "default_preprocessing",
    "default_postprocessing",
    "default_projection"
]


def default_preprocessing(
        in_data: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        gamma: torch.Tensor,
        mask: torch.Tensor = None,
        **conditioning
) -> torch.Tensor:
    """
    The default preprocessing function for the sampler. If conditioning data is
    given, `in_data` will be concatenated with the conditioning data.
    """
    if conditioning:
        return torch.cat((in_data, *conditioning.values()), dim=1)
    else:
        return in_data


def default_postprocessing(
        prediction: torch.Tensor,
        in_data: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        gamma: torch.Tensor,
        mask: torch.Tensor = None,
        **conditioning
) -> torch.Tensor:
    """
    Default function to post-process the prediction of the neural network. This
    default function simply returns the prediction.
    """
    return prediction


def default_projection(
        prediction: torch.Tensor,
        in_data: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        gamma: torch.Tensor,
        mask: torch.Tensor = None,
        **conditioning
) -> torch.Tensor:
    """
    A default function projecting  `in_data` with the prediction to the
    (cleaned) state. This function applies Tweedie's formula, Efron (2011), for
    the projection.
    """
    return (in_data-sigma*prediction) / alpha
