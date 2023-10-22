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


class UnconditionalModule(LightningModule):
    def __init__(
            self,
            denoising_network: torch.nn.Module,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
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
        lr : float, default = 1E-3
            The learning rate during training.
        sampler : {ddm_dynamical.sampler.sampler.BaseSampler, None}, default = None
            The sampler defines the sampling process during the prediction step.
        """
        super().__init__()
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
        norm_gamma_t = self.scheduler.get_normalized_gamma(sampled_time)
        gamma_t = self.scheduler.denormalize_gamma(norm_gamma_t)
        var_t = torch.sigmoid(-gamma_t)

        # Estimate prediction
        latent = self.encoder(data)
        noised_latent = (1 - var_t).sqrt() * latent + var_t.sqrt() * noise
        prediction = self.denoising_network(
            noised_latent, normalized_gamma=norm_gamma_t.view(-1, 1), mask=mask,
            **{k: v for k, v in batch.items() if k not in ["data", "mask"]}
        )

        # Estimate losses
        weighting = -self.scheduler.get_gamma_deriv(sampled_time)
        error_noise = (prediction - noise).pow(2)
        total_loss = utils.masked_average(
            weighting * error_noise, mask
        )

        # Log losses
        self.log(f'{prefix}/loss', total_loss, batch_size=batch_size,
                 prog_bar=True)

        # Estimate auxiliary data loss
        state = (noised_latent-var_t.sqrt()*prediction) / (1-var_t).sqrt()
        data_abs_err = (state-data).abs()
        data_loss = utils.masked_average(data_abs_err, mask)
        self.log(f'{prefix}/data_loss', data_loss, prog_bar=False,
                 batch_size=batch_size)
        return {
            "loss": total_loss,
            "sampled_time": sampled_time,
            "diff_sq_err": error_noise
        }

    def on_train_batch_end(
            self,
            outputs: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        density_target = utils.masked_average(
            outputs["diff_sq_err"].detach(), batch["mask"],
            dim=tuple(range(1, outputs["diff_sq_err"].dim()))
        )
        self.scheduler.update(
            outputs["sampled_time"].squeeze(), density_target
        )
        print(self.scheduler.get_normalized_gamma(torch.linspace(0, 1, 11, device=self.device)))
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
        if self.weight_decay is None:
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                betas=(0.9, 0.99)
            )
        else:
            diffusion_params = list(
                self.denoising_network.parameters()
            ) + list(
                self.encoder.parameters()
            ) + list(
                self.decoder.parameters()
            )
            optimizer = torch.optim.AdamW([
                dict(
                    params=diffusion_params,
                    lr=self.lr,
                    betas=(0.9, 0.99),
                    weight_decay=self.weight_decay
                ),
                dict(
                    params=list(self.scheduler.parameters()),
                    lr=1E-3,
                    betas=(0.9, 0.99),
                    weight_decay=0.
                )
            ]
            )
        return optimizer
