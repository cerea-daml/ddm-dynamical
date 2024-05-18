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
            final_cleaning: bool = True,
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
        self.final_cleaning = final_cleaning
        self.heun = heun
        self.grad_scale = grad_scale

    def reconstruct(
            self,
            in_tensor: torch.Tensor,
            n_steps: int = 250,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        latent = super().reconstruct(in_tensor, n_steps, **conditioning)
        if self.final_cleaning:
            return self.cleaning_step(latent, **conditioning)
        return latent

    def cleaning_step(
            self, latent: torch.Tensor, **conditioning
    ) -> torch.Tensor:
        gamma_s = self.scheduler(
            torch.zero(1, 1, device=latent.device, dtype=latent.dtype)
        )
        var_s = torch.sigmoid(-gamma_s)
        alpha_s = (1 - var_s).sqrt()
        sigma_s = var_s.sqrt()
        prediction = self.estimate_prediction(
            in_data=latent,
            alpha=alpha_s,
            sigma=sigma_s,
            gamma=gamma_s,
            **conditioning
        )
        return self.param(
            prediction=prediction,
            in_data=latent,
            alpha=alpha_s,
            sigma=sigma_s,
            gamma=gamma_s,
            **conditioning
        )

    def forward(
            self,
            in_data: torch.Tensor,
            step: torch.Tensor,
            **conditioning: Dict[str, Any]
    ) -> torch.Tensor:
        # Estimate coefficients
        prev_step = step - 1 / self.timesteps
        gamma_t = self.scheduler(1 - step)
        gamma_s = self.scheduler(1 - prev_step)
        var_t = torch.sigmoid(-gamma_t)
        var_s = torch.sigmoid(-gamma_s)
        alpha_t = (1 - var_t).sqrt()
        alpha_s = (1 - var_s).sqrt()
        sigma_t = var_t.sqrt()
        sigma_s = var_s.sqrt()

        # Coefficients for exploding
        sigma_tilde_t = gamma_t.exp() ** (-0.5)
        sigma_tilde_s = gamma_s.exp() ** (-0.5)
        dt = sigma_tilde_t - sigma_tilde_s

        # Estimate predictions
        prediction = self.estimate_prediction(
            in_data=in_data,
            alpha=alpha_s,
            sigma=sigma_s,
            gamma=gamma_s,
            **conditioning
        )
        denoised = self.param(
            prediction=prediction,
            in_data=in_data,
            alpha=alpha_s,
            sigma=sigma_s,
            gamma=gamma_s,
            **conditioning
        )

        # Estimate grad in exploding
        in_exploded = in_data / alpha_s

        # Epsilon scaling
        grad = (in_exploded - denoised) / sigma_tilde_s / self.grad_scale
        out_exploded = in_exploded + grad * dt
        out_preserved = out_exploded * alpha_t
        if self.heun:
            # Heun step
            prediction = self.estimate_prediction(
                in_data=out_preserved,
                alpha=alpha_t,
                sigma=sigma_t,
                gamma=gamma_t,
                **conditioning
            )
            denoised = self.param(
                prediction=prediction,
                in_data=out_preserved,
                alpha=alpha_t,
                sigma=sigma_t,
                gamma=gamma_t,
                **conditioning
            )
            grad_2 = (out_exploded - denoised) / sigma_tilde_t / self.grad_scale
            total_grad = (grad + grad_2) / 2

            out_exploded = in_exploded + total_grad * dt
            out_preserved = out_exploded * alpha_t
        return out_preserved
