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
from typing import Any, Tuple, Optional

# External modules
from pytorch_lightning import LightningModule

import torch

# Internal modules


logger = logging.getLogger(__name__)


class VDMDiscreteModule(LightningModule):
    def __init__(
            self,
            denoising_network: torch.nn.Module,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
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
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.sampler = sampler
        self.save_hyperparameters(
            ignore=["denoising_network", "encoder", "decoder", "scheduler",
                    "sampler"]
        )

    def forward(
            self, in_tensor: torch.Tensor, time_tensor: torch.Tensor,
            *tensors: torch.Tensor,
    ):
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
        return self.denoising_network(in_tensor, time_tensor, *tensors)

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
            sampled_time: torch.Tensor
    ) -> torch.Tensor:
        gamma_t = self.scheduler.get_gamma(sampled_time)
        gamma_s = self.scheduler.get_gamma(sampled_time-1/self.timesteps)
        weighting = torch.expm1(gamma_t - gamma_s)
        error_noise = (prediction - noise).pow(2)
        loss_diff = 0.5 * self.timesteps * weighting * error_noise
        return loss_diff.sum()/prediction.size(0)

    def get_prior_loss(
            self,
            latent: torch.Tensor,
    ) -> torch.Tensor:
        gamma_1 = self.scheduler.get_gamma(
            torch.ones(1, dtype=latent.dtype, device=latent.device)
        )
        var_1 = torch.sigmoid(gamma_1)
        loss_latent = 0.5 * torch.sum(
            (1-var_1) * torch.square(latent) + var_1 - torch.log(var_1) - 1.,
        )
        return loss_latent/latent.size(0)

    def get_recon_loss(
            self,
            batch: torch.Tensor,
            latent: torch.Tensor,
            noise: torch.Tensor,
    ) -> torch.Tensor:
        gamma_0 = self.scheduler.get_gamma(
            torch.zeros(1, dtype=batch.dtype, device=batch.device)
        )
        var_0 = torch.sigmoid(gamma_0)
        x_hat = (1-var_0).sqrt() * latent + var_0.sqrt() * noise
        log_likelihood = self.decoder.log_likelihood(x_hat, batch)
        return -log_likelihood

    def estimate_loss(
            self,
            data: torch.Tensor,
            *tensors: torch.Tensor,
            prefix: str = "train",
    ) -> torch.Tensor:
        # Sampling
        noise = torch.randn_like(data)
        sampled_time = self.sample_time(data)

        # Evaluate scheduler
        gamma_t = self.scheduler.get_gamma(sampled_time)
        var_t = torch.sigmoid(gamma_t)

        # Estimate prediction
        latent = self.encoder(data)
        noised_latent = (1 - var_t).sqrt() * latent + var_t.sqrt() * noise
        prediction = self.denoising_network(
            noised_latent, sampled_time.view(-1, 1), *tensors
        )

        # Estimate losses
        loss_recon = self.get_recon_loss(data, latent, noise).mean()
        loss_diff = self.get_diff_loss(prediction, noise, sampled_time).mean()
        loss_prior = self.get_prior_loss(latent).mean()
        total_loss = loss_recon + loss_diff + loss_prior

        self.log_dict({
            f"{prefix}/loss_recon": loss_recon,
            f"{prefix}/loss_diff": loss_diff,
            f"{prefix}/loss_prior": loss_prior,
        })
        self.log(f'{prefix}/loss', total_loss)

        # Estimate auxiliary data loss
        state = (noised_latent-var_t.sqrt()*prediction) / (1-var_t).sqrt()
        data_loss = (state-data).abs().mean()
        self.log(f'{prefix}/data_loss', data_loss, on_step=False, on_epoch=True,
                 prog_bar=False)
        return total_loss
    
    def training_step(
            self,
            data: Any,
            batch_idx: int
    ) -> Any:
        if isinstance(data, torch.Tensor):
            total_loss = self.estimate_loss(data, prefix='train')
        else:
            total_loss = self.estimate_loss(*data, prefix='train')
        return total_loss

    def validation_step(
            self,
            data: Any,
            batch_idx: int
    ) -> Any:
        if isinstance(data, torch.Tensor):
            total_loss = self.estimate_loss(data, prefix='val')
        else:
            total_loss = self.estimate_loss(*data, prefix='val')
        return total_loss

    def test_step(
            self,
            data: Any,
            batch_idx: int
    ) -> Any:
        if isinstance(data, torch.Tensor):
            total_loss = self.estimate_loss(data, prefix='test')
        else:
            total_loss = self.estimate_loss(*data, prefix='test')
        return total_loss

    def predict_step(
            self,
            data: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        if self.sampler is None:
            raise ValueError("To predict with diffusion model, "
                             "please set sampler!")
        if isinstance(data, torch.Tensor):
            sample = self.sampler.sample(data.shape)
            sample = self.decoder(sample)
        else:
            sample = self.sampler.sample(data[0].shape, *data[1:])
            sample = self.decoder(sample, *data[1:])
        return sample

    def configure_optimizers(
            self
    ) -> "torch.optim.Optimizer":
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        return optimizer
