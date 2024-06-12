#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 25/03/2024
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Callable, Dict, Any

# External modules
import torch

# Internal modules
from .sampler import BaseSampler

main_logger = logging.getLogger(__name__)


class HeunSampler(BaseSampler):
    def __init__(
            self,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 20,
            denoising_network: torch.nn.Module = None,
            pre_func: Callable = None,
            post_func: Callable = None,
            prior_sampler: Callable = None,
            grad_scale: float = 1.0,
            param: Callable = None,
            heun: bool = True,
            gamma_min: float = -15.,
            gamma_max: float = 15.,
            pbar: bool = False
    ):
        """
        Heun sampler from Karras et al. (2022), Alg. 2.
        """
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
            pbar=pbar
        )
        self.heun = heun
        self.grad_scale = grad_scale

    def forward(
            self,
            in_data: torch.Tensor,
            curr_stats: Dict[str, torch.Tensor],
            next_stats: Dict[str, torch.Tensor],
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Estimate coefficients
        # Coefficients for exploding
        sigma_tilde_curr = curr_stats["gamma"].exp() ** (-0.5)
        sigma_tilde_next = next_stats["gamma"].exp() ** (-0.5)
        dt = sigma_tilde_next - sigma_tilde_curr

        # Estimate predictions
        prediction = self.estimate_prediction(
            in_data=in_data,
            alpha=curr_stats["alpha"],
            sigma=curr_stats["sigma"],
            gamma=curr_stats["gamma"],
            **conditioning
        )
        denoised = self.param(
            prediction=prediction,
            in_data=in_data,
            alpha=curr_stats["alpha"],
            sigma=curr_stats["sigma"],
            gamma=curr_stats["gamma"],
            **conditioning
        )

        # Estimate grad in exploding
        in_exploded = in_data / curr_stats["alpha"]

        # Epsilon scaling
        grad = (in_exploded - denoised) / sigma_tilde_curr / self.grad_scale
        out_exploded = in_exploded + grad * dt
        out_preserved = out_exploded * next_stats["alpha"]

        if self.heun and next_stats["step"] > 0:
            # Heun step
            prediction = self.estimate_prediction(
                in_data=out_preserved,
                alpha=next_stats["alpha"],
                sigma=next_stats["sigma"],
                gamma=next_stats["gamma"],
                **conditioning
            )
            denoised = self.param(
                prediction=prediction,
                in_data=out_preserved,
                alpha=next_stats["alpha"],
                sigma=next_stats["sigma"],
                gamma=next_stats["gamma"],
                **conditioning
            )
            grad_2 = (out_exploded - denoised) / sigma_tilde_next / self.grad_scale
            total_grad = (grad + grad_2) / 2

            out_exploded = in_exploded + total_grad * dt
            out_preserved = out_exploded * next_stats["alpha"]
        return out_preserved
