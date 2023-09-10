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
from typing import Tuple

# External modules
import torch
from torch.func import grad

# Internal modules
from .vdm_discrete import VDMDiscreteModule


logger = logging.getLogger(__name__)


class VDMContinuousModule(VDMDiscreteModule):
    def __init__(
            self,
            denoising_network: torch.nn.Module,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            lr: float = 1E-3,
            sampler: "ddm_dynamical.sampler.sampler.BaseSampler" = None
    ) -> None:
        """
        A module to train an unconditional Gaussian denoising diffusion model
        for a given network and noise scheduler. A
        sampler can be additionally set to use the denoising diffusion model for
        prediction.

        Parameters
        ----------
        denoising_network : torch.nn.Module
            This denoising neural network will be trained by this module
        scheduler : ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler
            This noise scheduler defines the signal to noise ratio for the time
            steps during training.
        lr : float, default = 1E-3
            The learning rate during training.
        sampler : {ddm_dynamical.sampler.sampler.BaseSampler, None}, default = None
            The sampler defines the sampling process during the prediction step.
        """
        super().__init__(
            denoising_network=denoising_network,
            encoder=encoder,
            decoder=decoder,
            scheduler=scheduler,
            timesteps=None,
            lr=lr,
            sampler=sampler
        )

    def sample_time(
            self,
            template_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples time indices as tensor between [0, 1].

        Parameters
        ----------
        template_tensor : torch.Tensor
            The template tensor with `n` dimensions and shape (batch size, *).

        Returns
        -------
        sampled_time : torch.Tensor
            The time tensor sampled for each sample independently. The resulting
            shape is (batch size, *) with `n` dimensions filled by ones. The
            tensor lies on the same device as the input.
        """
        time_shape = torch.Size(
            [template_tensor.size(0)] + [1, ] * (template_tensor.ndim-1)
        )
        time_shift = torch.randn(
            1, dtype=template_tensor.dtype, device=template_tensor.device
        )
        sampled_time = torch.linspace(
            0, 1, template_tensor.size(0),
            dtype=template_tensor.dtype, device=template_tensor.device
        )
        sampled_time = (time_shift+sampled_time) % 1
        sampled_time = sampled_time.reshape(time_shape)
        return sampled_time

    def get_diff_loss(
            self,
            prediction: torch.Tensor,
            noise: torch.Tensor,
            sampled_time: torch.Tensor,
    ) -> torch.Tensor:
        sampled_time = torch.nn.Parameter(sampled_time)
        weighting = grad(
            lambda x: self.scheduler.get_gamma(x).sum()
        )(sampled_time)
        error_noise = (prediction - noise).pow(2)
        return 0.5 * weighting * error_noise
