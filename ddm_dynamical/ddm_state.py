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
from typing import Any

# External modules
from pytorch_lightning import LightningModule

import torch
import torch.nn.functional as F

# Internal modules


logger = logging.getLogger(__name__)


class StateDDMModule(LightningModule):
    def __init__(
            self,
            denoising_network: torch.nn.Module,
            head: "ddm_dynamical.head_param.head.HeadParam",
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 1000,
            lr: float = 1E-3,
            sampler: "ddm_dynamical.sampler.sampler.BaseSampler" = None
    ) -> None:
        """
        A module to train an unconditional Gaussian denoising diffusion model
        for a given network, head parameterization, and noise scheduler. A
        sampler can be additionally set to use the denoising diffusion model for
        prediction.

        Parameters
        ----------
        denoising_network : torch.nn.Module
            This denoising neural network will be trained by this module
        head : ddm_dynamical.head_param.head.HeadParam
            This head parameterization defines the output of the neural network
            and the loss function used during training.
        scheduler : ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler
            This noise scheduler defines the signal to noise ratio for the time
            steps during training.
        timesteps : int, default = 1000
            The number of time steps used during training.
        lr : float, default = 1E-3
            The learning rate during training.
        sampler : {ddm_dynamical.sampler.sampler.BaseSampler, None}, default = None
            The sampler defines the sampling process during the prediction step.
        """
        super().__init__()
        self.timesteps = timesteps
        self.scheduler = scheduler
        self.head = head
        self.denoising_network = denoising_network
        self.lr = lr
        self.sampler = sampler
        self.save_hyperparameters(
            ignore=["denoising_network", "head", "scheduler", "sampler"]
        )

    def forward(self, in_tensor: torch.Tensor, idx_time: torch.Tensor):
        """
        Apply the denoising network without the head parameterization for one
        single step.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The noised input for the neural network.
        idx_time : torch.LongTensor
            The time index feeded as additional input to the neural network

        Returns
        -------
        output_tensor : torch.Tensor
            The neural network output without the applied head parameterization.
            The output has the same shape as the in_tensor.
        """
        return self.denoising_network(in_tensor, idx_time)

    def sample_time(
            self,
            template_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples time indices as long tensor between [1, `self.timestep`].

        Parameters
        ----------
        template_tensor : torch.Tensor
            The template tensor with `n` dimensions and shape (batch size, *).

        Returns
        -------
        sampled_time : torch.LongTensor
            The time tensor sampled for each sample independently. The resulting
            shape is (batch size, *) with `n` dimensions filled by ones. The
            tensor lies on the same device as the input.
        """
        time_shape = torch.Size(
            [template_tensor.size(0)] + [1, ] * (template_tensor.ndim-1)
        )
        sampled_time = torch.randint(
            1, self.timesteps+1, time_shape,
            device=template_tensor.device
        ).long()
        return sampled_time

    def diffuse(
            self,
            in_data: torch.Tensor,
            alpha_sqrt: Any,
            sigma: Any,
    ) -> [torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(in_data)
        noised_data = alpha_sqrt * in_data + sigma * noise
        return noised_data, noise

    def estimate_loss(
            self,
            batch: torch.Tensor,
            prefix: str = "train"
    ) -> torch.Tensor:
        sampled_time = self.sample_time(batch)
        alpha_sqrt = self.scheduler.get_alpha(sampled_time).sqrt()
        sigma = self.scheduler.get_sigma(sampled_time)
        noised_data, noise = self.diffuse(batch, alpha_sqrt, sigma)
        prediction = self.denoising_network(
            noised_data, sampled_time.view(-1, 1)
        )
        head_loss = self.head(
            state=batch,
            noise=noise,
            prediction=prediction,
            alpha_sqrt=alpha_sqrt,
            sigma=sigma
        )
        self.log(f'{prefix}/loss', head_loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        denoised_batch = self.head.get_state(
            latent_state=noised_data,
            prediction=prediction,
            alpha_sqrt=alpha_sqrt,
            sigma=sigma
        )
        data_loss = F.l1_loss(denoised_batch, batch)
        self.log(f'{prefix}/data_loss', data_loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        return head_loss
    
    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix='train')
        return total_loss

    def validation_step(
            self,
            batch: torch.Tensor,
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix='val')
        return total_loss

    def test_step(
            self,
            batch: torch.Tensor,
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix='test')
        return total_loss

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        if self.sampler is None:
            raise ValueError("To predict with diffusion model, "
                             "please set sampler!")
        return self.sampler.sample(batch.shape)

    def configure_optimizers(
            self
    ) -> "torch.optim.Optimizer":
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        return optimizer
