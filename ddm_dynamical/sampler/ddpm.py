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

# External modules
import torch.nn

# Internal modules
from .sampler import BaseSampler

main_logger = logging.getLogger(__name__)


class DDPMSampler(BaseSampler):
    def forward(
            self,
            in_data: torch.Tensor,
            step: torch.Tensor,
            mask: torch.Tensor = None,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        # Estimate coefficients from scheduler
        prev_step = step-1/self.timesteps
        norm_gamma_t = self.scheduler.get_normalized_gamma(step)
        gamma_t = self.scheduler.denormalize_gamma(norm_gamma_t)
        gamma_s = self.scheduler(step-1/self.timesteps)
        var_t = torch.sigmoid(-gamma_t)
        alpha_s_sq = torch.sigmoid(gamma_s)

        sigma_t = var_t.sqrt()
        alpha_t_sq = 1-var_t
        alpha_dash_t_sq = alpha_t_sq / alpha_s_sq
        alpha_t = alpha_t_sq.sqrt()
        alpha_s = alpha_s_sq.sqrt()
        alpha_dash_t = alpha_dash_t_sq.sqrt()

        # Estimate factors
        latent_factor = alpha_dash_t*(1-alpha_s_sq)/(1-alpha_t_sq)
        state_factor = alpha_s*(1-alpha_dash_t_sq)/(1-alpha_t_sq)
        noise_factor = (
                (1-alpha_dash_t_sq)*(1-alpha_s_sq)/(1-alpha_t_sq)
        ).sqrt()

        # Estimate tensors
        time_tensor = torch.ones(
            in_data.size(0), 1, device=in_data.device, dtype=in_data.dtype
        ) * step
        prediction = self.denoising_network(
            in_data, normalized_gamma=norm_gamma_t, mask=mask, **conditioning
        )
        state = self.proj_func(
            prediction=prediction,
            in_data=in_data,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            time_tensor=time_tensor,
            mask=mask,
            **conditioning
        )

        if prev_step > 0:
            noise = torch.randn_like(in_data)
            state = latent_factor * in_data + state_factor * state + \
                    noise_factor * noise
        return state
