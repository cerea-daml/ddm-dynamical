#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27/08/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import lightning.pytorch as pl

from ddm_dynamical.encoder import GaussianEncoder
from ddm_dynamical.decoder import GaussianDecoder
from ddm_dynamical.sampler import HeunSampler
from ddm_dynamical.scheduler import BinarizedScheduler, EDMSamplingScheduler
from ddm_dynamical.weighting import ExponentialWeighting

# Internal modules
from data_modules import UnconditionalStateDataModule
from networks import UNeXt
from unconditional import UnconditionalModule

main_logger = logging.getLogger(__name__)


def train_model():
    pl.seed_everything(42)

    data_module = UnconditionalStateDataModule(
        "data/", batch_size=1024
    )
    main_logger.info("Initialized data")

    denoising_network = UNeXt()
    encoder = GaussianEncoder()
    decoder = GaussianDecoder()
    model = UnconditionalModule(
        denoising_network=denoising_network,
        encoder=encoder,
        decoder=decoder,
        scheduler=BinarizedScheduler(gamma_min=-10, gamma_max=10),
        weighting=ExponentialWeighting(),
        lr=3E-4,
        sampler=HeunSampler(
            scheduler=EDMSamplingScheduler(gamma_min=-10, gamma_max=10),
            denoising_network=denoising_network,
            timesteps=40,
            heun=True
        )
    )
    main_logger.info("Initialized model")

    trainer = pl.Trainer(
        max_epochs=200,
        benchmark=False,
        deterministic=True,
        accelerator="auto"
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    train_model()
