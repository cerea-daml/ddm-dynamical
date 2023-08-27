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
            idx_time: int = 1000
    ) -> torch.Tensor:
        curr_idx_time = torch.round(idx_time * self.step_mul).long()
        prev_idx_time = torch.round((idx_time - 1) * self.step_mul).long()
        alpha_t = self.noise_scheduler.get_alpha(curr_idx_time)
        alpha_s = self.noise_scheduler.get_alpha(prev_idx_time)
        alpha_dash_t = alpha_t / alpha_s
        alpha_sqrt_t = alpha_t.sqrt()
        alpha_sqrt_s = alpha_s.sqrt()
        alpha_dash_sqrt_t = alpha_dash_t.sqrt()
        sigma_t = self.noise_scheduler.get_sigma(curr_idx_time)

        latent_factor = alpha_dash_sqrt_t*(1-alpha_s)/(1-alpha_t)
        state_factor = alpha_sqrt_s*(1-alpha_dash_t)/(1-alpha_t)
        noise_factor = ((1-alpha_dash_t)*(1-alpha_s)/(1-alpha_t)).sqrt()

        time_tensor = torch.full(
            torch.Size(
                [in_data.shape[0]] + [1, ] * (in_data.ndim - 1)
            ), curr_idx_time, device=in_data.device,
            dtype=in_data.dtype, layout=in_data.layout
        )
        prediction = self.denoising_model(in_data, time_tensor)
        noise = torch.randn_like(in_data)

        return latent_factor * in_data + \
            state_factor * self.head.get_state(
                in_data, prediction=prediction,
                alpha_sqrt=alpha_sqrt_t, sigma=sigma_t
            ) + noise_factor * noise
