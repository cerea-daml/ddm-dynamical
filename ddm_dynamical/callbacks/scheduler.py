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

# External modules
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torch


# Internal modules

main_logger = logging.getLogger(__name__)


class EvaluateSchedulerCallback(Callback):
    def __init__(self, n_steps: int = 1001):
        super().__init__()
        self.timesteps = torch.linspace(0, 1, n_steps)

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        if trainer.logger is not None:
            curr_param = next(pl_module.parameters())
            self.timesteps.to(curr_param)
            gamma_t = pl_module.scheduler.get_gamma(self.timesteps)
            alpha_t = torch.sigmoid(-gamma_t).sqrt()

            fig, ax = plt.subplots()
            ax.plot(self.timesteps.cpu(), -gamma_t.cpu())
            ax.set_xlim(0, 1)
            ax.set_xlabel("Time")
            ax.set_ylabel("log SNR")

            trainer.logger.log_image(
                "scheduler/logSNR", [fig]
            )

            fig, ax = plt.subplots()
            ax.plot(self.timesteps.cpu(), alpha_t.cpu())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal")

            trainer.logger.log_image(
                "scheduler/signal", [fig]
            )
            plt.close("all")
