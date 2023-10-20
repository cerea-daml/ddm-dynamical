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
from ddm_dynamical.scheduler.ddpm_scheduler import DDPMScheduler


logging.basicConfig(level=logging.DEBUG)


class TestDDPMScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = DDPMScheduler()
        self.timesteps = torch.rand(12)
        self.timesteps[0] = 0.
        self.timesteps[-1] = 1.

    def test_returns_approximation_with_swapped_time(self):
        true_gamma = -torch.log(torch.expm1(
            self.scheduler.shift +
            self.scheduler.scale * self.timesteps.pow(2)
        ))
        returned_gamma = self.scheduler._estimate_gamma(self.timesteps)
        torch.testing.assert_close(returned_gamma, true_gamma)

    def test_normalized_returns_normalized(self):
        true_gamma = -torch.log(torch.expm1(
            self.scheduler.shift +
            self.scheduler.scale * self.timesteps.pow(2)
        ))
        true_norm = (true_gamma-true_gamma[-1])/(true_gamma[0]-true_gamma[-1])
        gamma = self.scheduler._estimate_gamma(self.timesteps)
        returned_norm = self.scheduler.normalize_gamma(gamma)
        torch.testing.assert_close(returned_norm, true_norm)

    def test_forward_returns_in_limits(self):
        true_gamma = -torch.log(torch.expm1(
            self.scheduler.shift +
            self.scheduler.scale * self.timesteps.pow(2)
        ))
        true_norm = (true_gamma-true_gamma[-1])/(true_gamma[0]-true_gamma[-1])
        true_gamma = self.scheduler.gamma_min + \
            (self.scheduler.gamma_max-self.scheduler.gamma_min) * true_norm
        returned_gamma = self.scheduler(self.timesteps)
        torch.testing.assert_close(returned_gamma, true_gamma)
