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
from typing import Any, Dict

# External modules
from pytorch_lightning import LightningModule

import torch

# Internal modules
from ddm_dynamical import utils


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
            weight_decay: float = None,
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
        self.weight_decay = weight_decay
        self.save_hyperparameters(
            ignore=["denoising_network", "encoder", "decoder", "scheduler",
                    "sampler"]
        )

    def forward(
            self, in_tensor: torch.Tensor, time_tensor: torch.Tensor,
            mask: torch.Tensor = None, **conditioning: torch.Tensor
    ):
        """
        Predict the noise given the noised input tensor and the time.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The noised input for the neural network.
        time_tensor : torch.Tensor
            The continuous input time [0, 1].
        mask : torch.Tensor, default = None
            The mask [0, 1] indicating which values are valid. Default is None
            for cases where the neural network shouldn't be masked.
        conditioning: torch.Tensor
            Additional tensors as keyword arguments. These tensors are
            additional conditioning information, useable within the neural
            network.

        Returns
        -------
        output_tensor : torch.Tensor
            The predicted noise.
        """
        return self.denoising_network(
            in_tensor, time_tensor, mask, **conditioning
        )

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
        gamma_t = self.scheduler(sampled_time)
        gamma_s = self.scheduler(sampled_time-1/self.timesteps)
        weighting = torch.expm1(gamma_t - gamma_s)
        error_noise = (prediction - noise).pow(2)
        loss_diff = 0.5 * self.timesteps * weighting * error_noise
        return loss_diff

    def get_prior_loss(
            self,
            latent: torch.Tensor,
    ) -> torch.Tensor:
        gamma_1 = self.scheduler(
            torch.ones(1, dtype=latent.dtype, device=latent.device)
        )
        var_1 = torch.sigmoid(gamma_1)
        loss_latent = 0.5 * (
                (1-var_1) * torch.square(latent)
                + var_1 - torch.log(var_1) - 1.
        )
        return loss_latent

    def get_recon_loss(
            self,
            data: torch.Tensor,
            latent: torch.Tensor,
            noise: torch.Tensor
    ) -> torch.Tensor:
        gamma_0 = self.scheduler(
            torch.zeros(1, dtype=data.dtype, device=data.device)
        )
        var_0 = torch.sigmoid(gamma_0)
        x_hat = (1-var_0).sqrt() * latent + var_0.sqrt() * noise
        log_likelihood = self.decoder.log_likelihood(x_hat, data)
        return -log_likelihood

    def estimate_loss(
            self,
            batch: torch.Tensor,
            prefix: str = "train",
    ) -> torch.Tensor:
        # Pop data and mask out of batch
        data = batch.pop("data")
        batch_size = data.size(0)
        mask = batch.pop("mask", None)

        # Sampling
        noise = torch.randn_like(data)
        sampled_time = self.sample_time(data)

        # Evaluate scheduler
        gamma_t = self.scheduler(sampled_time)
        var_t = torch.sigmoid(gamma_t)

        # Estimate prediction
        latent = self.encoder(data)
        noised_latent = (1 - var_t).sqrt() * latent + var_t.sqrt() * noise
        prediction = self.denoising_network(
            noised_latent, time_tensor=sampled_time.view(-1, 1), mask=mask,
            **batch
        )

        # Estimate losses
        loss_recon = utils.masked_average(
            self.get_recon_loss(data, latent, noise), mask
        )
        loss_diff = utils.masked_average(
            self.get_diff_loss(prediction, noise, sampled_time), mask
        )
        loss_prior = utils.masked_average(
            self.get_prior_loss(latent), mask
        )
        total_loss = loss_recon + loss_diff + loss_prior

        # Log losses
        self.log_dict({
            f"{prefix}/loss_recon": loss_recon,
            f"{prefix}/loss_diff": loss_diff,
            f"{prefix}/loss_prior": loss_prior,
        }, batch_size=batch_size)
        self.log(f'{prefix}/loss', total_loss, batch_size=batch_size,
                 prog_bar=True)

        # Estimate auxiliary data loss
        state = (noised_latent-var_t.sqrt()*prediction) / (1-var_t).sqrt()
        data_abs_err = (state-data).abs()
        data_loss = utils.masked_average(data_abs_err, mask)
        self.log(f'{prefix}/data_loss', data_loss, prog_bar=False,
                 batch_size=batch_size)
        return total_loss
    
    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix="train")
        return total_loss

    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix="val")
        return total_loss

    def test_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Any:
        total_loss = self.estimate_loss(batch, prefix="test")
        return total_loss

    def predict_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        if self.sampler is None:
            raise ValueError("To predict with diffusion model, "
                             "please set sampler!")
        data = batch.pop("data")
        mask = batch.pop("mask", None)
        sample = self.sampler.sample(data.shape, mask=mask, **batch)
        sample = self.decoder(sample, mask=mask)
        return sample

    def configure_optimizers(
            self
    ) -> "torch.optim.Optimizer":
        if self.weight_decay is None:
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                betas=(0.9, 0.99)
            )
        else:
            diffusion_params = list(self.denoising_network.parameters())
            other_params = list(self.scheduler.parameters()) + \
                           list(self.decoder.parameters())
            optimizer = torch.optim.AdamW([
                dict(
                    params=diffusion_params,
                    lr=self.lr,
                    betas=(0.9, 0.99),
                    weight_decay=self.weight_decay
                ),
                dict(
                    params=other_params,
                    lr=self.lr,
                    betas=(0.9, 0.99),
                    weight_decay=0.
                )
            ]
            )
        return optimizer
