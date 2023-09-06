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

# Internal modules


logger = logging.getLogger(__name__)


class VDMDiscreteModule(LightningModule):
    def __init__(
            self,
            denoising_network: torch.nn.Module,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 1000,
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
        self.denoising_network = denoising_network
        self.lr = lr
        self.sampler = sampler
        self.recon_logvar = torch.nn.Parameter(torch.zeros(5, 1, 1))
        self.save_hyperparameters(
            ignore=["denoising_network", "scheduler", "sampler"]
        )

    def forward(self, in_tensor: torch.Tensor, time_tensor: torch.Tensor):
        """
        Predict the noise given the noised input tensor and the time.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The noised input for the neural network.
        time_tensor : torch.Tensor
            The continuous input time [0, 1].

        Returns
        -------
        output_tensor : torch.Tensor
            The predicted noise.
            The output has the same shape as the in_tensor.
        """
        return self.denoising_network(in_tensor, time_tensor)

    def sample_time(
            self,
            template_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples time indices as long tensor between [1, timesteps]/timesteps.

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
        sampled_time = torch.randint(
            1, self.timesteps+1, time_shape,
            dtype=template_tensor.dtype,
            device=template_tensor.device
        ) / self.timesteps
        return sampled_time

    def get_diff_loss(
            self,
            prediction: torch.Tensor,
            noise: torch.Tensor,
            weighting: torch.Tensor
    ) -> torch.Tensor:
        error_noise = (prediction - noise).pow(2).sum()
        loss_diff = 0.5 * self.timesteps * weighting * error_noise
        return loss_diff/prediction.size(0)

    def get_latent_loss(
            self,
            batch: torch.Tensor,
    ) -> torch.Tensor:
        gamma_1 = self.scheduler.get_gamma(
            torch.ones(1, dtype=batch.dtype, device=batch.device)
        )
        var_1 = torch.sigmoid(gamma_1)
        loss_latent = 0.5 * torch.sum(
            (1-var_1) * torch.square(batch) + var_1 - torch.log(var_1) - 1.,
        )
        return loss_latent/batch.size(0)

    def get_recon_loss(
            self,
            batch: torch.Tensor,
            noise: torch.Tensor,
    ) -> torch.Tensor:
        gamma_0 = self.scheduler.get_gamma(
            torch.zeros(1, dtype=batch.dtype, device=batch.device)
        )
        var_0 = torch.sigmoid(gamma_0)
        x_hat = (1-var_0).sqrt() * batch + var_0.sqrt() * noise
        data_part = ((x_hat-batch).pow(2)/self.recon_logvar.exp())
        logvar_part = self.recon_logvar*torch.ones_like(batch)
        const_part = (2 * torch.pi * torch.ones_like(batch)).log()
        loss_recon = 0.5 * (data_part+logvar_part+const_part).sum()
        return loss_recon/batch.size(0)

    def estimate_loss(
            self,
            batch: torch.Tensor,
            prefix: str = "train"
    ) -> torch.Tensor:
        # Sampling
        noise = torch.randn_like(batch)
        sampled_time = self.sample_time(batch)

        # Evaluate scheduler
        gamma_t = self.scheduler.get_gamma(sampled_time)
        gamma_s = self.scheduler.get_gamma(sampled_time-1/self.timesteps)
        var_t = torch.sigmoid(gamma_t)
        weighting = torch.expm1(gamma_t-gamma_s)

        # Estimate prediction
        noised_data = (1 - var_t).sqrt() * batch + var_t.sqrt() * noise
        prediction = self.denoising_network(
            noised_data, sampled_time.view(-1, 1)
        )

        # Estimate losses
        loss_recon = self.get_recon_loss(batch, noise).mean()
        loss_diff = self.get_diff_loss(prediction, noise, weighting).mean()
        loss_latent = self.get_latent_loss(batch).mean()
        total_loss = loss_recon + loss_diff + loss_latent

        self.log_dict({
            f"{prefix}/loss_recon": loss_recon,
            f"{prefix}/loss_diff": loss_diff,
            f"{prefix}/loss_latent": loss_latent,
        })
        self.log(f'{prefix}/loss', total_loss)

        # Estimate auxiliary data loss
        state = (noised_data-var_t.sqrt()*prediction) / (1-var_t).sqrt()
        data_loss = (state-batch).abs().mean()
        self.log(f'{prefix}/data_loss', data_loss, on_step=False, on_epoch=True,
                 prog_bar=False)
        return total_loss
    
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
