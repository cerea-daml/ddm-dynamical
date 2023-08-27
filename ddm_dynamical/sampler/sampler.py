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
            head: "dyn_ddim.head_param.head.HeadParam",
            timesteps: int = 250,
            denoising_model: torch.nn.Module = None,
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.head = head
        self.timesteps = timesteps
        self.noise_scheduler = scheduler
        self.step_mul = self.noise_scheduler.timesteps / timesteps

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
            start_time: int = 100
    ) -> torch.Tensor:
        denoised_tensor = in_tensor
        for idx_time in reversed(range(1, start_time+1)):
            denoised_tensor = self(denoised_tensor, torch.tensor(idx_time))
        return denoised_tensor

    def forward(
            self,
            in_data: torch.Tensor,
            idx_time: int = 1000
    ) -> torch.Tensor:
        pass
