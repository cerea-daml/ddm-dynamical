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
            in_tensor: torch.Tensor,
            step: torch.Tensor
    ) -> torch.Tensor:
        # Estimate coefficients from scheduler
        prev_step = step-1/self.timesteps
        gamma_t = self.scheduler.get_gamma(step)
        gamma_s = self.scheduler.get_gamma(prev_step)
        var_t = torch.sigmoid(gamma_t)
        alpha_t = (1-var_t)
        alpha_s = torch.sigmoid(-gamma_s)
        alpha_dash_t = alpha_t / alpha_s
        alpha_sqrt_s = alpha_s.sqrt()
        alpha_dash_sqrt_t = alpha_dash_t.sqrt()

        # Estimate factors
        latent_factor = alpha_dash_sqrt_t*(1-alpha_s)/(1-alpha_t)
        state_factor = alpha_sqrt_s*(1-alpha_dash_t)/(1-alpha_t)
        noise_factor = ((1-alpha_dash_t)*(1-alpha_s)/(1-alpha_t)).sqrt()

        # Estimate tensors
        time_tensor = torch.ones(
            in_tensor.size(0), device=in_tensor.device, dtype=in_tensor.dtype
        ) * step
        prediction = self.denoising_model(in_tensor, time_tensor)
        state = (in_tensor-var_t.sqrt()*prediction) / (1-var_t).sqrt()

        if prev_step > 0:
            noise = torch.randn_like(in_tensor)
            state = latent_factor * in_tensor + state_factor * state + \
                    noise_factor * noise
        return state
