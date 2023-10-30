#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 30/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Callable

# External modules
import torch.nn

# Internal modules
from .sampler import BaseSampler

main_logger = logging.getLogger(__name__)


class KDiffusionSampler(BaseSampler):
    def __init__(
            self,
            k_func: Callable,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 250,
            denoising_network: torch.nn.Module = None,
            proj_func: Callable = None,
            pbar: bool = False
    ):
        super().__init__(
            scheduler=scheduler,
            timesteps=timesteps,
            denoising_network=denoising_network,
            proj_func=proj_func,
            pbar=pbar
        )
        self.k_func = k_func

    @staticmethod
    def input_to_exploding(
            in_tensor: torch.Tensor,
            gamma: torch.Tensor
    ) -> torch.Tensor:
        scaling_factor = (1 + 1/gamma.exp()).sqrt()
        return in_tensor * scaling_factor

    def denoise_step(
            self,
            in_state: torch.Tensor,
            sigma_tilde: torch.Tensor,
            mask: torch.Tensor = None,
            **conditioning
    ) -> torch.Tensor:
        var_exploded = sigma_tilde[0]**2
        var_preserved = var_exploded / (var_exploded+1)
        alpha = (1-var_preserved).sqrt()
        sigma = var_preserved.sqrt()
        gamma = torch.log(1/var_exploded)
        scaled_in_state = in_state / (var_exploded + 1).sqrt()
        prediction = self.estimate_prediction(
            scaled_in_state, gamma=gamma, mask=mask, **conditioning
        )
        return self.proj_func(
            prediction,
            scaled_in_state,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            mask=mask,
            **conditioning
        )

    def reconstruct(
            self,
            in_tensor: torch.Tensor,
            n_steps: int = 250,
            mask: torch.Tensor = None,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        time_steps = torch.linspace(
            0, 1, self.timesteps+1,
            device=in_tensor.device, dtype=in_tensor.dtype,
            layout=in_tensor.layout
        )[0:n_steps+1]
        time_steps = torch.flip(time_steps, dims=(0, ))
        gammas = self.scheduler(time_steps)
        sigma_tildes = gammas.exp()**(-0.5)
        in_tensor_exploded = self.input_to_exploding(in_tensor, gammas[0])
        return self.k_func(
            self.denoise_step,
            in_tensor_exploded,
            sigma_tildes,
            extra_args=dict(mask=mask, **conditioning),
            disable=False
        )
