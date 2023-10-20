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
from ddm_dynamical.scheduler.noise_scheduler import NoiseScheduler


logging.basicConfig(level=logging.DEBUG)


class MockScheduler(NoiseScheduler):
    def _estimate_gamma(
            self, timesteps: torch.Tensor
    ) -> torch.Tensor:
        return timesteps * -5


class TestNoiseScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = MockScheduler()
        self.timesteps = torch.rand(10)

    def test_normalize_returns_normalized_gamma(self):
        gamma = self.scheduler._estimate_gamma(
            self.timesteps
        )
        norm_gamma = self.scheduler.normalize_gamma(gamma)
        torch.testing.assert_close(norm_gamma, 1-self.timesteps)

    def test_forward_wo_normalize_returns_gamma(self):
        gamma = self.scheduler._estimate_gamma(
            self.timesteps
        )
        self.scheduler.normalize = False
        returned = self.scheduler(self.timesteps)
        torch.testing.assert_close(returned, gamma)

    def test_forward_w_normalize_returns_in_bounds(self):
        norm_gamma = 1-self.timesteps
        target = self.scheduler.gamma_min + (self.scheduler.gamma_max-self.scheduler.gamma_min) * norm_gamma
        self.scheduler.normalize = True
        returned = self.scheduler(self.timesteps)
        torch.testing.assert_close(returned, target)

    def test_forward_w_uses_gamma_min_max(self):
        self.scheduler.gamma_min = -6.5
        self.scheduler.gamma_max = 12.125
        norm_gamma = 1-self.timesteps
        target = self.scheduler.gamma_min + (self.scheduler.gamma_max-self.scheduler.gamma_min) * norm_gamma
        self.scheduler.normalize = True
        returned = self.scheduler(self.timesteps)
        torch.testing.assert_close(returned, target)

    def test_forward_is_diff_for_gamma(self):
        self.scheduler = MockScheduler(
            gamma_min=torch.nn.Parameter(torch.ones(1,) * -5),
            gamma_max=torch.nn.Parameter(torch.ones(1,) * 12.5),
            normalize=True
        )
        self.assertIsNone(self.scheduler.gamma_min.grad)
        self.assertIsNone(self.scheduler.gamma_max.grad)

        returned = self.scheduler(self.timesteps)
        (returned-1).sum().backward()

        self.assertIsNotNone(self.scheduler.gamma_min.grad)
        self.assertIsNotNone(self.scheduler.gamma_max.grad)
