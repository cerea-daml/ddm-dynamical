#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 29/06/2023
# Created for 2022_ddim_for_attractors
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict, Any, Callable

# External modules
import torch.nn

# Internal modules
from .sampler import BaseSampler

main_logger = logging.getLogger(__name__)


class DDPMSampler(BaseSampler):
    def forward(
            self,
            in_data: torch.Tensor,
            curr_stats: Dict[str, torch.Tensor],
            next_stats: Dict[str, torch.Tensor],
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Estimate tensors
        prediction = self.estimate_prediction(
            in_data=in_data,
            alpha=curr_stats["alpha"],
            sigma=curr_stats["sigma"],
            gamma=curr_stats["gamma"],
            **conditioning
        )
        if next_stats["step"] > 0:
            # Gamma definition swapped to VDM paper!
            # Alpha^2 = sigmoid(gamma), sigma^2 = sigmoid(-gamma)
            factor = torch.expm1(curr_stats["gamma"] - next_stats["gamma"])

            noise = self.param.get_noise(
                prediction=prediction,
                in_data=in_data,
                alpha=curr_stats["alpha"],
                sigma=curr_stats["sigma"],
                gamma=curr_stats["gamma"],
                **conditioning
            )
            scale = (
                next_stats["alpha"] + 1E-9
            ) / (
                curr_stats["alpha"] + 1E-9
            )
            mean = scale * (in_data + curr_stats["sigma"] * factor * noise)

            scale_added_noise = next_stats["sigma"] * torch.sqrt(-factor)
            state = mean + scale_added_noise * torch.randn_like(mean)
        else:
            state = self.param(
                prediction=prediction,
                in_data=in_data,
                alpha=curr_stats["alpha"],
                sigma=curr_stats["sigma"],
                gamma=curr_stats["gamma"],
                **conditioning
            )
        return state
