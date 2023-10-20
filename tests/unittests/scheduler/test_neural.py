#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 20/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import unittest
import logging

# External modules
import torch

# Internal modules
from ddm_dynamical.scheduler.nn_scheduler import NNScheduler


logging.basicConfig(level=logging.DEBUG)


class TestNNScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = NNScheduler()
        self.timesteps = torch.rand(12)
        self.timesteps[:2] = torch.arange(2)

    def test_neural_network_returns_linear_at_init(self):
        slope = self.scheduler.gamma_min-self.scheduler.gamma_max
        true_gamma = self.scheduler.gamma_max + slope * self.timesteps
        returned_gamma = self.scheduler._estimate_gamma(self.timesteps)
        torch.testing.assert_close(returned_gamma, true_gamma)

    def test_nn_monotonical_increasing(self):
        self.scheduler.branch_factor.data[:] = 1000.
        returned_gamma = self.scheduler._estimate_gamma(
            torch.linspace(0, 1, 101)
        )
        for i, v in enumerate(returned_gamma[1:]):
            self.assertLess(v, returned_gamma[i])

    def test_nn_after_update_monotonical(self):
        self.scheduler.branch_factor.data[:] = 1000.
        slope = self.scheduler.gamma_min-self.scheduler.gamma_max
        true_gamma = self.scheduler.gamma_max + slope * self.timesteps
        returned_gamma = self.scheduler._estimate_gamma(self.timesteps)
        optimizer = torch.optim.SGD(self.scheduler.parameters(), lr=1E-2)
        optimizer.zero_grad()
        error = (returned_gamma-true_gamma).pow(2).mean()
        error.backward()
        optimizer.step()

        with torch.no_grad():
            new_gamma = self.scheduler._estimate_gamma(
                torch.linspace(0, 1, 101)
            )
        for i, v in enumerate(new_gamma[1:]):
            self.assertLess(v, new_gamma[i])
