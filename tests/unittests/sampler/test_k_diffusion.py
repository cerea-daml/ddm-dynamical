#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 30/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import unittest
import logging
from math import exp

# External modules
import torch
from k_diffusion.sampling import sample_euler

# Internal modules
from ddm_dynamical.sampler.k_diffusion import KDiffusionSampler
from ddm_dynamical.scheduler import LinearScheduler


logging.basicConfig(level=logging.DEBUG)


def dummy_model(x, normalized_gamma, mask):
    return torch.ones_like(x)


class TestKDiffusion(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = KDiffusionSampler(
            k_func=sample_euler,
            scheduler=LinearScheduler(),
            denoising_network=dummy_model,
        )

    def test_scale_input_scales_input_to_exploding(self):
        gamma = torch.tensor(-5.)
        variance = torch.sigmoid(-gamma)
        alpha = (1-variance).sqrt()
        sigma = variance.sqrt()
        state = torch.randn(4, 8, 32, 32)
        noise = torch.randn_like(state)
        in_tensor = alpha * state + sigma * noise

        sigma_tilde = (1/gamma.exp()).sqrt()
        exploded_tensor = state + sigma_tilde * noise

        returned_tensor = self.sampler.input_to_exploding(
            in_tensor, gamma
        )
        torch.testing.assert_close(returned_tensor, exploded_tensor)
