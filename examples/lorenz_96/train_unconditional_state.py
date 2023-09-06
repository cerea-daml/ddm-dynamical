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
import pytorch_lightning as pl

from data_modules import UnconditionalStateDataModule
from ddm_dynamical.sampler import DDIMSampler
from ddm_dynamical.scheduler import DDPMScheduler
from ddm_dynamical.vdm_discrete import VDMDiscreteModule
# Internal modules
from networks import UNeXt

main_logger = logging.getLogger(__name__)


def train_model():
    pl.seed_everything(42)

    data_module = UnconditionalStateDataModule(
        "data/", batch_size=1024
    )
    main_logger.info("Initialized data")

    denoising_network = UNeXt()
    scheduler = DDPMScheduler()
    model = VDMDiscreteModule(
        denoising_network=denoising_network,
        scheduler=scheduler,
        lr=3E-4,
        sampler=DDIMSampler(
            scheduler=scheduler,
            denoising_model=denoising_network,
            timesteps=100
        )
    )
    main_logger.info("Initialized model")

    trainer = pl.Trainer(
        max_epochs=200,
        benchmark=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    train_model()
