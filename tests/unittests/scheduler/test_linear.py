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
        self.timesteps = torch.rand(10)

    def test_gamma_returns_linear_from_max_to_min(self):
        slope = self.scheduler.gamma_min-self.scheduler.gamma_max
        true_gamma = self.scheduler.gamma_max + slope * self.timesteps
        returned_gamma = self.scheduler._estimate_gamma(self.timesteps)
        torch.testing.assert_close(returned_gamma, true_gamma)

    def test_normalized_gamma_returns_1_0_gamma(self):
        true_normalized = 1-self.timesteps
        gamma = self.scheduler._estimate_gamma(self.timesteps)
        returned_normalized = self.scheduler.get_normalized_gamma(gamma)
        torch.testing.assert_close(returned_normalized, true_normalized)

    def test_forward_returns_always_same(self):
        true_gamma = self.scheduler._estimate_gamma(self.timesteps)
        self.scheduler.normalize = False
        unnorm_gamma = self.scheduler(self.timesteps)
        torch.testing.assert_close(unnorm_gamma, true_gamma)
        self.scheduler.normalize = True
        norm_gamma = self.scheduler(self.timesteps)
        torch.testing.assert_close(norm_gamma, true_gamma)


