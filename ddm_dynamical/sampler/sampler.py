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
from typing import Callable

# External modules
import torch
from tqdm.autonotebook import tqdm

# Internal modules
from .defaults import *
from ddm_dynamical.utils import normalize_gamma


logger = logging.getLogger(__name__)


class BaseSampler(torch.nn.Module):
    def __init__(
            self,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 250,
            denoising_network: torch.nn.Module = None,
            pre_func: Callable = None,
            post_func: Callable = None,
            proj_func: Callable = None,
            gamma_min: float = -15.,
            gamma_max: float = 15.,
            pbar: bool = True,
    ):
        super().__init__()
        self.denoising_network = denoising_network
        self.pre_func = pre_func or default_preprocessing
        self.post_func = post_func or default_postprocessing
        self.proj_func = proj_func or eps_projection
        self.timesteps = timesteps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.scheduler = scheduler
        self.pbar = pbar

    def generate_prior_sample(
            self,
            sample_shape=torch.Size([])
    ) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        template_tensor = next(self.denoising_network.parameters())
        prior_sample = torch.randn(
            sample_shape, device=template_tensor.device,
            dtype=template_tensor.dtype, layout=template_tensor.layout
        )
        return prior_sample

    def estimate_prediction(
            self,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            mask: torch.Tensor,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        in_tensor = self.pre_func(
            in_data=in_data,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            mask=mask,
            **conditioning
        )
        norm_gamma = torch.ones(
            in_tensor.size(0), 1, device=in_tensor.device, dtype=in_tensor.dtype
        ) * normalize_gamma(gamma, self.gamma_min, self.gamma_max)
        prediction = self.denoising_network(
            in_tensor, normalized_gamma=norm_gamma, mask=mask, **conditioning
        )
        prediction = self.post_func(
            prediction=prediction,
            in_data=in_data,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            mask=mask,
            **conditioning
        )
        return prediction

    @torch.no_grad()
    def sample(
            self,
            sample_shape=torch.Size([]),
            mask: torch.Tensor = None,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        prior_sample = self.generate_prior_sample(sample_shape)
        denoised_data = self.reconstruct(
            prior_sample, self.timesteps, mask=mask, **conditioning
        )
        return denoised_data

    def reconstruct(
            self,
            in_tensor: torch.Tensor,
            n_steps: int = 250,
            mask: torch.Tensor = None,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        denoised_tensor = in_tensor
        time_steps = torch.linspace(
            0, 1, self.timesteps+1,
            device=in_tensor.device, dtype=in_tensor.dtype,
            layout=in_tensor.layout
        )[1:n_steps+1]
        if self.pbar:
            time_steps = tqdm(reversed(time_steps), total=n_steps, leave=False)
        for step in time_steps:
            denoised_tensor = self(
                denoised_tensor, step, mask=mask, **conditioning
            )
        return denoised_tensor

    def forward(
            self,
            in_data: torch.Tensor,
            step: torch.Tensor,
            mask: torch.Tensor = None,
            **conditioning: torch.Tensor
    ) -> torch.Tensor:
        pass
