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
from typing import Callable, Dict, Any

# External modules
import torch

# Internal modules
from .sampler import BaseSampler


logger = logging.getLogger(__name__)


class DDIMSampler(BaseSampler):
    def __init__(
            self,
            scheduler: "dyn_ddim.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 250,
            denoising_network: torch.nn.Module = None,
            pre_func: Callable = None,
            post_func: Callable = None,
            prior_sampler: Callable = None,
            param: Callable = None,
            ddpm: bool = False,
            eta: float = 0.,
            gamma_min: float = -15.,
            gamma_max: float = 15.,
            pbar: bool = True,
            sample_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(
            scheduler=scheduler,
            timesteps=timesteps,
            denoising_network=denoising_network,
            pre_func=pre_func,
            post_func=post_func,
            prior_sampler=prior_sampler,
            param=param,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            pbar=pbar,
            sample_kwargs=sample_kwargs
        )
        self.ddpm = ddpm
        self.eta = eta

    def _estimate_stoch_level(self, alpha_sq_curr, alpha_sq_next):
        stoch_level_sq = (
             (1 - alpha_sq_next + 1E-9) / (1 - alpha_sq_curr + 1E-9)
             * (1 - (alpha_sq_curr + 1E-9) / (alpha_sq_next + 1E-9))
        )
        if self.ddpm:
            stoch_level = torch.sqrt(
                1 - (alpha_sq_curr + 1E-9) / (alpha_sq_next + 1E-9)
            )
        else:
            stoch_level_sq *= self.eta**2
            stoch_level = stoch_level_sq.sqrt()
        det_level = (1 - alpha_sq_next - stoch_level_sq).sqrt()
        return det_level, stoch_level

    def forward(
            self,
            in_data: torch.Tensor,
            curr_stats: Dict[str, torch.Tensor],
            next_stats: Dict[str, torch.Tensor],
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Estimate predictions
        prediction = self.estimate_prediction(
            in_data=in_data,
            alpha=curr_stats["alpha"],
            sigma=curr_stats["sigma"],
            gamma=curr_stats["gamma"],
            **conditioning
        )
        state = self.param(
            prediction=prediction,
            in_data=in_data,
            alpha=curr_stats["alpha"],
            sigma=curr_stats["sigma"],
            gamma=curr_stats["gamma"],
            **conditioning
        )
        if next_stats["step"] == 0:
            return state

        noise_level = self._estimate_stoch_level(
            alpha_sq_curr=curr_stats["alpha_sq"],
            alpha_sq_next=next_stats["alpha_sq"]
        )
        noise = self.param.get_noise(
            prediction=prediction,
            in_data=in_data,
            alpha=curr_stats["alpha"],
            sigma=curr_stats["sigma"],
            gamma=curr_stats["gamma"],
            **conditioning
        )
        added_noise = torch.randn_like(in_data)
        return (
            next_stats["alpha"] * state
            + noise_level[0] * noise
            + noise_level[1] * added_noise
        )
