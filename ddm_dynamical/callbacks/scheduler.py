#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11/09/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Any

# External modules
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch


# Internal modules

main_logger = logging.getLogger(__name__)


class EvaluateSchedulerCallback(Callback):
    def __init__(self, n_steps: int = 1001):
        super().__init__()
        self.timesteps = torch.linspace(0, 1, n_steps)

    def on_train_epoch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.logger is not None:
            self.timesteps = self.timesteps.to(pl_module.device)
            gamma = pl_module.scheduler(self.timesteps)
            density = pl_module.scheduler.get_density(gamma)
            alpha = torch.sigmoid(gamma).sqrt()

            fig, ax = plt.subplots()
            ax.plot(self.timesteps.cpu(), gamma.cpu())
            ax.set_xlim(0, 1)
            ax.set_xlabel("Time")
            ax.set_ylabel("log SNR")

            trainer.logger.log_image(
                "scheduler/logSNR", [fig]
            )

            fig, ax = plt.subplots()
            ax.plot(gamma.cpu(), density.cpu())
            ax.set_xlabel("log SNR")
            ax.set_ylabel("Density")

            trainer.logger.log_image(
                "scheduler/density", [fig]
            )

            fig, ax = plt.subplots()
            ax.plot(self.timesteps.cpu(), alpha.cpu())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal")

            trainer.logger.log_image(
                "scheduler/signal", [fig]
            )
            plt.close("all")
