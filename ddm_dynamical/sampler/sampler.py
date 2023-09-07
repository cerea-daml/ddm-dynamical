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

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class BaseSampler(torch.nn.Module):
    def __init__(
            self,
            scheduler: "dyn_ddim.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 250,
            denoising_model: torch.nn.Module = None,
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.timesteps = timesteps
        self.scheduler = scheduler

    def generate_prior_sample(
            self,
            sample_shape=torch.Size([])
    ) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        template_tensor = next(self.denoising_model.parameters())
        prior_sample = torch.randn(
            sample_shape, device=template_tensor.device,
            dtype=template_tensor.dtype, layout=template_tensor.layout
        )
        return prior_sample

    @torch.no_grad()
    def sample(
            self,
            sample_shape=torch.Size([])
    ) -> torch.Tensor:
        prior_sample = self.generate_prior_sample(sample_shape)
        denoised_data = self.reconstruct(prior_sample, self.timesteps)
        return denoised_data

    def reconstruct(
            self,
            in_tensor: torch.Tensor,
            n_steps: int = 250
    ) -> torch.Tensor:
        denoised_tensor = in_tensor
        time_steps = torch.linspace(
            0, 1, self.timesteps+1,
            device=in_tensor.device, dtype=in_tensor.dtype,
            layout=in_tensor.layout
        )[1:n_steps+1]
        for step in reversed(time_steps):
            denoised_tensor = self(denoised_tensor, step)
        return denoised_tensor

    def forward(
            self,
            in_data: torch.Tensor,
            step: torch.Tensor
    ) -> torch.Tensor:
        pass
