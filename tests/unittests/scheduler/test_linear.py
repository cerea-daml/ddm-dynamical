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
from ddm_dynamical.scheduler.linear_scheduler import LinearScheduler


logging.basicConfig(level=logging.DEBUG)


class TestLinearScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = LinearScheduler()

    def test_linear_returns_gamma_from_max_to_min(self):
        timesteps = torch.rand(1024)
        correct_gamma = (1-timesteps) * 20 - 10
        returned_gamma = self.scheduler(timesteps)
        torch.testing.assert_close(returned_gamma, correct_gamma)

        # Chagne gamma min and gamma max
        scale = 30
        shift = -20
        self.scheduler.gamma_min = shift
        self.scheduler.gamma_max = scale+shift
        correct_gamma = (1 - timesteps) * scale + shift
        returned_gamma = self.scheduler(timesteps)
        torch.testing.assert_close(returned_gamma, correct_gamma)

    def test_density_constant(self):
        timesteps = torch.rand(1024)
        correct = torch.ones(1024) / (
                self.scheduler.gamma_max-self.scheduler.gamma_min
        )

        returned = self.scheduler.get_density(timesteps)
        torch.testing.assert_close(returned, correct)
