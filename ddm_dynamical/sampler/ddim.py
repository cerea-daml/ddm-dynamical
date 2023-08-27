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
            head: "dyn_ddim.head_param.head.HeadParam",
            timesteps: int = 250,
            denoising_model: torch.nn.Module = None,
            ddpm: bool = False,
            eta: float = 0.,
    ):
        super().__init__(
            scheduler=scheduler,
            head=head,
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
            idx_time: int = 1000
    ) -> torch.Tensor:
        curr_idx_time = torch.round(idx_time * self.step_mul).long()
        prev_idx_time = torch.round((idx_time-1) * self.step_mul).long()
        alpha_t = self.noise_scheduler.get_alpha(curr_idx_time)
        alpha_s = self.noise_scheduler.get_alpha(prev_idx_time)
        alpha_sqrt_t = alpha_t.sqrt()
        alpha_sqrt_s = alpha_s.sqrt()
        sigma_t = self.noise_scheduler.get_sigma(curr_idx_time)
        sigma_s = self.noise_scheduler.get_sigma(prev_idx_time)

        time_tensor = torch.full(
            torch.Size(
                [in_data.shape[0]] + [1, ] * (in_data.ndim - 1)
            ), curr_idx_time, device=in_data.device,
            dtype=in_data.dtype, layout=in_data.layout
        )
        prediction = self.denoising_model(in_data, time_tensor)

        state = self.head.get_state(
            in_data, prediction=prediction,
            alpha_sqrt=alpha_sqrt_t, sigma=sigma_t
        )
        if idx_time > 1:
            noise = self.head.get_noise(
                in_data, prediction=prediction,
                alpha_sqrt=alpha_sqrt_t, sigma=sigma_t
            )
            if self.ddpm or self.eta > 0:
                noise_level = self._estimate_stoch_level(
                    alpha_t=alpha_t, alpha_s=alpha_s
                )
                state = alpha_sqrt_s * state \
                        + noise_level[0] * noise \
                        + noise_level[1] * torch.randn_like(noise)
            else:
                state = alpha_sqrt_s * state + sigma_s * noise
        return state
