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
from lightning.pytorch import LightningModule

import torch

# Internal modules
from ddm_dynamical import utils
from ddm_dynamical.weighting.elbo import ELBOWeighting


logger = logging.getLogger(__name__)


class UnconditionalModule(LightningModule):
    def __init__(
            self,
            denoising_network: torch.nn.Module,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            weighting=None,
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
        encoder : torch.nn.Module
            The encoding function that transforms the input into latent space.
        decoder : torch.nn.Module
            The decoding function that translates from latent space back to
            state space.
        scheduler : ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler
            This noise scheduler defines the signal to noise ratio for the
            training. The training scheduler can be different from the sampling
            scheduler.
        weighting : torch.nn.Module, default = None
            The weighting function that determines the weighting of the
            different loss terms in the diffusion loss. If None, then a constant
            weighting, corresponding to the original ELBO is used.
        lr : float, default = 1E-3
            The learning rate during training.
        sampler : {ddm_dynamical.sampler.sampler.BaseSampler, None}, default = None
            The sampler defines the sampling process during the prediction step.
        """
        super().__init__()
        self.scheduler = scheduler
        self.denoising_network = denoising_network
        if weighting is None:
            weighting = ELBOWeighting()
        self.weighting = weighting
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.sampler = sampler
        self.save_hyperparameters(
            ignore=["denoising_network", "encoder", "decoder", "scheduler",
                    "weighting", "sampler"]
        )

    def forward(
            self, in_tensor: torch.Tensor, normalized_gamma: torch.Tensor,
            mask: torch.Tensor = None, **conditioning: torch.Tensor
    ):
        """
        Predict the noise given the noised input tensor and the time.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The noised input for the neural network.
        normalized_gamma : torch.Tensor
            The log signal to noise ratio, normalized to [1, 0].
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
            in_tensor, normalized_gamma=normalized_gamma, mask=mask,
            **conditioning
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

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor],
            prefix: str = "train",
    ) -> Dict[str, torch.Tensor]:
        # Pop data and mask out of batch
        data = batch["data"]
        batch_size = data.size(0)
        mask = batch["mask"]

        # Sampling
        noise = torch.randn_like(data)
        sampled_time = self.sample_time(data)

        # Evaluate scheduler
        gamma = self.scheduler(sampled_time)
        noise_var = torch.sigmoid(-gamma)

        # Estimate prediction
        latent = self.encoder(data)
        noised_latent = (1 - noise_var).sqrt() * latent \
                        + noise_var.sqrt() * noise
        norm_gamma = self.scheduler.normalize_gamma(gamma.detach())
        prediction = self.denoising_network(
            noised_latent, normalized_gamma=norm_gamma.view(-1, 1), mask=mask,
            **{k: v for k, v in batch.items() if k not in ["data", "mask"]}
        )

        # Estimate losses
        weighting = self.weighting(gamma)
        density = self.scheduler.get_density(gamma)

        weighted_error = weighting * (prediction - noise).pow(2)
        total_loss = utils.masked_average(
            weighted_error/density, mask
        )

        # Log losses
        self.log(f'{prefix}/loss', total_loss, batch_size=batch_size,
                 prog_bar=True)

        # Estimate auxiliary data loss
        state = (noised_latent-noise_var.sqrt()*prediction) \
                / (1-noise_var).sqrt()
        data_abs_err = (state-data).abs()
        data_loss = utils.masked_average(data_abs_err, mask)
        self.log(f'{prefix}/data_loss', data_loss, prog_bar=False,
                 batch_size=batch_size)
        return {
            "loss": total_loss,
            "gamma": gamma,
            "weighted_error": weighted_error
        }

    def on_train_batch_end(
            self,
            outputs: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        density_target = utils.masked_average(
            outputs["weighted_error"].detach(), batch["mask"],
            dim=tuple(range(1, outputs["weighted_error"].dim()))
        )
        self.scheduler.update(
            outputs["gamma"].squeeze(), density_target
        )
        return outputs
    
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
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.99)
        )
        return optimizer
