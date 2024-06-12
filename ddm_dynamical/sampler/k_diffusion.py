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
            pre_func: Callable = None,
            post_func: Callable = None,
            prior_sampler: Callable = None,
            param: Callable = None,
            gamma_min: float = -15.,
            gamma_max: float = 15.,
            pbar: bool = True,
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
            pbar=pbar
        )
        self.k_func = k_func

    @staticmethod
    def input_to_exploding(
            in_tensor: torch.Tensor,
            gamma: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.sigmoid(gamma).sqrt()
        return in_tensor / alpha

    def denoise_step(
            self,
            in_state: torch.Tensor,
            sigma_tilde: torch.Tensor,
            **conditioning
    ) -> torch.Tensor:
        var_exploded = sigma_tilde[0]**2
        var_preserved = var_exploded / (var_exploded+1)
        alpha = (1-var_preserved).sqrt()
        sigma = var_preserved.sqrt()
        gamma = torch.log(1/var_exploded)
        scaled_in_state = in_state * alpha
        prediction = self.estimate_prediction(
            in_data=scaled_in_state,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            **conditioning
        )
        return self.param(
            prediction,
            scaled_in_state,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            **conditioning
        )

    def reconstruct(
            self,
            in_tensor: torch.Tensor,
            n_steps: int = 250,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        time_steps = torch.linspace(
            1, 0, self.timesteps+1,
            device=in_tensor.device, dtype=in_tensor.dtype,
            layout=in_tensor.layout
        )[-n_steps-1:]
        gammas = self.scheduler(time_steps)
        sigma_tildes = gammas.exp()**(-0.5)
        in_tensor_exploded = self.input_to_exploding(in_tensor, gammas[0])
        return self.k_func(
            self.denoise_step,
            in_tensor_exploded,
            sigma_tildes,
            extra_args=dict(**conditioning),
            disable=None if self.pbar else True
        )
