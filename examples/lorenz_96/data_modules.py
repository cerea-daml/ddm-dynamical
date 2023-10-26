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
import os.path

# External modules
from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader

# Internal modules
from datasets import *

main_logger = logging.getLogger(__name__)


class UnconditionalStateDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 16384,
            n_workers: int = 4,
            pin_memory: bool = True
    ):
        super().__init__()
        self._datasets = dict()
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        self._datasets = {
            "train": StateDataset(torch.load(
                os.path.join(self.data_path, "traj_train.pt"),
                map_location="cpu"
            )),
            "val": StateDataset(torch.load(
                os.path.join(self.data_path, "traj_val.pt"),
                map_location="cpu"
            )),
            "test": StateDataset(torch.load(
                os.path.join(self.data_path, "traj_test.pt"),
                map_location="cpu"
            )),
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._datasets["train"], batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers, pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._datasets["val"], batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._datasets["test"], batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=self.pin_memory
        )
