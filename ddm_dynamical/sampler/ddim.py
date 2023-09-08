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
from typing import Union

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
            denoising_model: torch.nn.Module = None,
            ddpm: bool = False,
            eta: float = 0.,
    ):
        super().__init__(
            scheduler=scheduler,
            timesteps=timesteps,
            denoising_model=denoising_model
        )
        self.ddpm = ddpm
        self.eta = eta

    def _estimate_stoch_level(self, alpha_t, alpha_s):
        if self.ddpm:
            sigma_t_2 = (1 - alpha_s + 1E-9) / (1 - alpha_t + 1E-9) \
                        * (1 - (alpha_t + 1E-9) / (alpha_s + 1E-9))
            det_level = (1 - alpha_s - sigma_t_2).sqrt()
            stoch_level = torch.sqrt(1 - (alpha_t + 1E-9) / (alpha_s + 1E-9))
        else:
            stoch_level = torch.sqrt((1-alpha_s+1E-9) / (1-alpha_t+1E-9)) \
                        * torch.sqrt(1-(alpha_t+1E-9) / (alpha_s+1E-9))
            stoch_level = self.eta * stoch_level
            det_level = (1-alpha_s-stoch_level.pow(2)).sqrt()
        return det_level, stoch_level

    def forward(
            self,
            in_data: torch.Tensor,
            step: torch.Tensor,
            *tensors: torch.Tensor,
    ) -> torch.Tensor:
        # Estimate coefficients
        prev_step = step-1/self.timesteps
        gamma_t = self.scheduler.get_gamma(step)
        gamma_s = self.scheduler.get_gamma(step-1/self.timesteps)
        var_t = torch.sigmoid(gamma_t)
        var_s = torch.sigmoid(gamma_s)
        alpha_sqrt_s = (1-var_s).sqrt()
        noise_level = self._estimate_stoch_level(
            alpha_t=1 - var_t, alpha_s=1 - var_s
        )

        # Estimate predictions
        time_tensor = torch.ones(
            in_data.size(0), 1, device=in_data.device, dtype=in_data.dtype
        ) * step
        prediction = self.denoising_model(in_data, time_tensor, *tensors)
        state = (in_data-var_t.sqrt()*prediction) / (1-var_t).sqrt()

        if prev_step > 0:
            noise = torch.randn_like(in_data)
            state = alpha_sqrt_s * state + noise_level[0] * prediction \
                    + noise_level[1] * noise
        return state
