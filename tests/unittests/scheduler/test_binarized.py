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
import numpy as np
from tqdm import tqdm

# Internal modules
from ddm_dynamical.scheduler.binarized_scheduler import BinarizedScheduler
from ddm_dynamical.scheduler.cosine_scheduler import CosineScheduler


logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(42)


class TestBinarizedScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = BinarizedScheduler()

    def test_values_can_be_set(self):
        values = torch.randn(100)**2
        new_scheduler = BinarizedScheduler(values=values)
        torch.testing.assert_close(new_scheduler.bin_values, values)

    def test_values_is_set_to_ones_if_none(self):
        new_scheduler = BinarizedScheduler(values=None)
        torch.testing.assert_close(new_scheduler.bin_values, torch.ones(100))

    def test_values_is_transferred_into_torch(self):
        values = np.random.normal(size=100)**2
        new_scheduler = BinarizedScheduler(values=values)
        torch.testing.assert_close(
            new_scheduler.bin_values, torch.tensor(values)
        )
        values = list(values)
        new_scheduler = BinarizedScheduler(values=values)
        torch.testing.assert_close(
            new_scheduler.bin_values, torch.tensor(values)
        )

    def test_bin_times_returns_bin_times(self):
        new_times = torch.rand(101)
        self.scheduler._bin_times = new_times
        torch.testing.assert_close(self.scheduler.bin_times, new_times)

    def test_update_times_updates_time(self):
        self.scheduler._bin_values = torch.randn(100)**2
        bin_times = torch.cumsum(
            torch.flip(self.scheduler._bin_values, dims=(0, )), dim=0
        )
        bin_times = torch.cat(
            (torch.zeros(1, device=bin_times.device), bin_times),
            dim=0
        )
        bin_times = torch.flip(bin_times / bin_times[-1], dims=(0, ))
        self.scheduler._update_times()
        torch.testing.assert_close(self.scheduler.bin_times, bin_times)

    def test_setting_new_values_updates_bin_times(self):
        values = torch.randn(100)**2
        bin_times = torch.cumsum(
            torch.flip(values, dims=(0, )), dim=0
        )
        bin_times = torch.cat(
            (torch.zeros(1, device=bin_times.device), bin_times),
            dim=0
        )
        bin_times = torch.flip(bin_times / bin_times[-1], dims=(0, ))
        self.scheduler.bin_values = values
        torch.testing.assert_close(self.scheduler.bin_times, bin_times)

    def test_bin_num_returns_correct_bin(self):
        bin_num = self.scheduler.get_bin_num(torch.tensor([-10]))
        self.assertEqual(bin_num, 0)
        bin_num = self.scheduler.get_bin_num(torch.tensor([-9.7]))
        self.assertEqual(bin_num, 1)
        bin_num = self.scheduler.get_bin_num(torch.tensor([9.7]))
        self.assertEqual(bin_num, 98)
        bin_num = self.scheduler.get_bin_num(torch.tensor([9.9]))
        self.assertEqual(bin_num, 99)
        bin_num = self.scheduler.get_bin_num(torch.tensor([10]))
        self.assertEqual(bin_num, 99)

    def test_time_left_returns_correct_idx(self):
        self.scheduler.bin_values = torch.randn(100)**2
        self.scheduler._update_times()

        idx_left = self.scheduler.get_left_time(torch.tensor([0.]))
        self.assertEqual(idx_left, 99)
        idx_left = self.scheduler.get_left_time(torch.tensor([1.]))
        self.assertEqual(idx_left, 0)

        test_time = torch.rand(128)
        smaller = self.scheduler.bin_times[:, None] > test_time[None, :]
        target_idx = (smaller.float().argmin(dim=0)-1).clamp(
            min=0, max=self.scheduler.n_bins-1
        )
        idx_left = self.scheduler.get_left_time(test_time)
        torch.testing.assert_close(idx_left, target_idx)

    def test_get_density(self):
        self.scheduler.bin_values = torch.ones(100)
        self.scheduler.bin_values += torch.randn(100) * 1E-5
        self.scheduler._update_times()

        test_time = torch.rand(1024)
        test_gamma = (1-test_time) * 20 - 10
        bin_num = self.scheduler.get_bin_num(test_gamma)

        target_density = self.scheduler.bin_values[bin_num]
        returned_density = self.scheduler.get_density(test_gamma)
        torch.testing.assert_close(returned_density, target_density)

    def test_forward_returns_gamma_uniform(self):
        self.scheduler.bin_values = torch.ones(100)
        self.scheduler._update_times()

        returned_gamma = self.scheduler(torch.tensor(1.))
        self.assertEqual(returned_gamma, self.scheduler.gamma_min)
        returned_gamma = self.scheduler(torch.tensor(0.))
        self.assertEqual(returned_gamma, self.scheduler.gamma_max)
        returned_gamma = self.scheduler(torch.tensor(0.5))
        self.assertEqual(returned_gamma, 0.)
        returned_gamma = self.scheduler(torch.tensor(0.995))
        self.assertEqual(returned_gamma, -9.9)


        timesteps = torch.rand(128)
        gamma_diff = self.scheduler.gamma_max-self.scheduler.gamma_min
        gamma_target = ((1-timesteps)-0.5)*gamma_diff

        returned_gamma = self.scheduler(timesteps)
        torch.testing.assert_close(returned_gamma, gamma_target)

    def test_forward_returns_gamma_random(self):
        self.scheduler.bin_values = torch.randn(100)**2
        self.scheduler._update_times()

        returned_gamma = self.scheduler(torch.tensor(1.))
        self.assertEqual(returned_gamma, self.scheduler.gamma_min)
        returned_gamma = self.scheduler(torch.tensor(0.))
        self.assertEqual(returned_gamma, self.scheduler.gamma_max)

        timesteps = torch.rand(1024)
        timesteps[:2] = torch.arange(2)
        idx_left = self.scheduler.get_left_time(timesteps)
        idx_right = idx_left+1
        time_left = self.scheduler.bin_times[idx_left]
        time_right = self.scheduler.bin_times[idx_right]
        gamma_left = self.scheduler.bin_limits[idx_left]
        gamma_right = self.scheduler.bin_limits[idx_right]

        time_diff = time_right-time_left
        gamma_diff = gamma_right-gamma_left
        dgamma_dtime = gamma_diff / time_diff

        delta_time = timesteps-time_left
        delta_gamma = dgamma_dtime * delta_time
        gamma_target = gamma_left + delta_gamma

        returned_gamma = self.scheduler(timesteps)
        torch.testing.assert_close(returned_gamma, gamma_target)

    def test_update_updates_correct_index(self):
        self.scheduler._bin_times = torch.linspace(
            1, 0, self.scheduler.n_bins+1
        )
        target_values = torch.randn(1) ** 2
        timesteps = torch.rand(1)
        gammas = (1-timesteps) * 20 - 10
        bin_num = self.scheduler.get_bin_num(gammas)
        mask = torch.zeros(self.scheduler.n_bins, dtype=torch.bool)
        mask[bin_num] = True
        old_values = self.scheduler.bin_values.clone()
        self.scheduler.update(gammas, target_values)
        new_values = self.scheduler.bin_values.clone()
        self.assertNotEqual(old_values[mask].item(), new_values[mask].item())
        torch.testing.assert_close(old_values[~mask], new_values[~mask])

    def test_update_updates_with_ema(self):
        self.scheduler.bin_values = torch.ones(100)
        self.scheduler._update_times()
        self.scheduler._bin_times = torch.linspace(
            1, 0, self.scheduler.n_bins+1
        )

        target_values = torch.randn(1024) ** 2
        timesteps = torch.rand(1024)
        gammas = (1-timesteps) * 20 - 10

        bin_num = self.scheduler.get_bin_num(gammas)

        n_idx = torch.zeros_like(self.scheduler.bin_values)
        n_idx = torch.scatter_add(
            n_idx, dim=0, index=bin_num, src=torch.ones_like(target_values)
        )
        factors = self.scheduler.ema_rate**n_idx
        scattered_rate = (1-factors[bin_num]) / n_idx[bin_num]

        scattered = factors * self.scheduler.bin_values
        scattered = torch.scatter_add(
            scattered, dim=0, index=bin_num, src=scattered_rate*target_values
        )

        self.scheduler.update(gammas, target_values)
        torch.testing.assert_close(self.scheduler.bin_values, scattered)

    def test_update_updates_time(self):
        self.scheduler.bin_values = torch.ones(100)
        self.scheduler._update_times()
        self.scheduler._bin_times = torch.linspace(
            1, 0, self.scheduler.n_bins+1
        )

        target_values = torch.randn(1024) ** 2
        timesteps = torch.rand(1024)
        gammas = (1-timesteps) * 20 - 10
        bin_num = self.scheduler.get_bin_num(gammas)
        n_idx = torch.zeros_like(self.scheduler.bin_values)
        n_idx = torch.scatter_add(
            n_idx, dim=0, index=bin_num, src=torch.ones_like(target_values)
        )
        factors = self.scheduler.ema_rate**n_idx
        scattered_rate = (1-factors[bin_num]) / n_idx[bin_num]
        scattered = factors * self.scheduler.bin_values
        scattered = torch.scatter_add(
            scattered, dim=0, index=bin_num, src=scattered_rate*target_values
        )
        bin_times = torch.cumsum(
            torch.flip(scattered, dims=(0, )), dim=0
        )
        bin_times = torch.cat(
            (torch.zeros(1, device=bin_times.device), bin_times),
            dim=0
        )
        bin_times = torch.flip(bin_times / bin_times[-1], dims=(0, ))
        self.scheduler.update(gammas, target_values)
        torch.testing.assert_close(self.scheduler.bin_times, bin_times)

    def test_binarized_can_approx_cosine(self):
        test_times = torch.linspace(0, 1, 1001)
        self.scheduler.ema_rate = 0.99
        target_scheduler = CosineScheduler()
        target_gammas = target_scheduler(test_times)
        old_gammas = self.scheduler(test_times)
        self.assertGreater((old_gammas-target_gammas).pow(2).mean(), 1E-5)
        for _ in range(1000):
            timesteps = torch.rand(1024)
            gamma = self.scheduler(timesteps)
            target_values = target_scheduler.get_density(gamma)
            self.scheduler.update(gamma, target_values)
        new_gammas = self.scheduler(test_times)
        self.assertLess((new_gammas-target_gammas).pow(2).mean(), 1E-5)
