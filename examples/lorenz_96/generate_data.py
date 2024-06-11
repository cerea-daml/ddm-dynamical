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
import os

# External modules
import torch

from tqdm import tqdm

# Internal modules
from lorenz_96 import Lorenz96
from integrator import RK4Integrator


logger = logging.getLogger(__name__)


def generate_data(
        data_path: str,
        dt: float = 0.05,
        n_grid: int = 40,
        forcing: float = 8.0,
        n_ensemble: int = 16,
        n_spinup: int = 4000,
        n_length: int = 100000,
        data_type: str = "train"
):
    model = Lorenz96(forcing=forcing)
    integrator = RK4Integrator(model, dt=dt)
    logger.info("Initialised the model")

    initial_state = torch.randn(n_ensemble, n_grid) * 0.001
    for _ in tqdm(range(n_spinup)):
        initial_state = integrator.integrate(initial_state)
    logger.info("Finished burn-in stage")

    trajectories = [initial_state.clone()]
    for _ in tqdm(range(n_length)):
        trajectories.append(integrator.integrate(trajectories[-1]))
    trajectories = torch.stack(trajectories, dim=1)
    logger.info("Generated the trajectories")

    shift = trajectories.mean()
    scale = trajectories.std()
    logger.info(f"mean: {shift}, std: {scale}")
    logger.info("Normalised the trajectories")

    torch.save(
        trajectories, os.path.join(data_path, f"traj_{data_type:s}.pt")
    )
    logger.info("Stored the trajectories")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    generate_data("data/", n_ensemble=16)
    logger.info("Finished training data")
    generate_data("data/", n_ensemble=1, data_type="val")
    logger.info("Finished validation data")
    generate_data("data/", n_ensemble=16, data_type="test")
    logger.info("Finished testing data")
