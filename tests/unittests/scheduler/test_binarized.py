#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 23/10/2023
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
from ddm_dynamical.scheduler.binarized_scheduler import BinarizedScheduler


logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(42)


class TestBinarizedScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = BinarizedScheduler()
        self.timesteps = torch.rand(16)
        self.timesteps[:2] = torch.arange(2)

    def test_get_gamma_deriv(self):
        self.scheduler.bin_values = torch.randn(100)**2
        left_bin = self.scheduler.get_left(self.timesteps)
        target_deriv = - 1 / self.scheduler.bin_values[left_bin]
        returned_deriv = self.scheduler.get_gamma_deriv(self.timesteps)
        torch.testing.assert_close(returned_deriv, target_deriv)

    def test_get_normalized_gamma(self):
        left_bin = self.scheduler.get_left(self.timesteps)
        self.scheduler.bin_values = torch.randn(100)**2
        integral = torch.cumsum(-1/self.scheduler.bin_values, dim=0)
        self.scheduler.bin_gamma[1:] = (integral[-1]-integral)/integral[-1]
        gamma_left = self.scheduler.bin_gamma[left_bin]
        gamma_right = self.scheduler.bin_gamma[left_bin+1]
        diff = gamma_right - gamma_left
        time_diff = self.timesteps - self.scheduler.bin_limits[left_bin]
        delta_gamma = diff * time_diff / self.scheduler.n_bins
        true_gamma = gamma_left+delta_gamma
        returned_gamma = self.scheduler.get_normalized_gamma(self.timesteps)
        torch.testing.assert_close(returned_gamma, true_gamma)

    def test_update_updates_with_ema(self):
        target_values = torch.randn(1024)**2
        timesteps = torch.rand(1024)
        idx_left = self.scheduler.get_left(timesteps)
        original = self.scheduler.bin_values.clone()

        n_idx = torch.zeros_like(original)
        n_idx = torch.scatter_add(
            n_idx, dim=0, index=idx_left, src=torch.ones_like(target_values)
        )
        factors = (1-self.scheduler.ema_rate)**n_idx
        scattered_rate = (1-factors[idx_left]) / n_idx[idx_left]

        scattered = factors * original
        scattered = torch.scatter_add(
            scattered, dim=0, index=idx_left, src=scattered_rate*target_values
        )
        self.scheduler.update(timesteps, target_values)
        torch.testing.assert_close(self.scheduler.bin_values, scattered)

    def test_update_updates_gammas(self):
        target_values = torch.randn(1024)**2
        timesteps = torch.rand(1024)
        idx_left = self.scheduler.get_left(timesteps)
        original = self.scheduler.bin_values.clone()

        n_idx = torch.zeros_like(original)
        n_idx = torch.scatter_add(
            n_idx, dim=0, index=idx_left, src=torch.ones_like(target_values)
        )
        factors = (1-self.scheduler.ema_rate)**n_idx
        scattered_rate = (1-factors[idx_left]) / n_idx[idx_left]

        scattered = factors * original
        scattered = torch.scatter_add(
            scattered, dim=0, index=idx_left, src=scattered_rate*target_values
        )
        derivative = -1/scattered
        integral = torch.cumsum(derivative, dim=0)
        correct_gamma = (integral[-1]-integral)/integral[-1]
        correct_gamma = torch.cat((torch.ones(1), correct_gamma)).abs()
        self.scheduler.update(timesteps, target_values)
        torch.testing.assert_close(self.scheduler.bin_gamma, correct_gamma)
