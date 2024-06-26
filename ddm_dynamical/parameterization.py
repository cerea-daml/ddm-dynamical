#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 09/03/2024
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


__all__ = [
    "EPSParam",
    "VParam",
    "DataParam",
    "FlowOTParam",
    "EDMParam",
    "ScoreParam"
]


class ScoreParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        target_score = -noise/sigma
        return sigma**2 * (target_score - prediction).pow(2)

    def get_noise(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        return -prediction * sigma

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        return 1 / alpha * (in_data + sigma * prediction)


class EPSParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        return (noise - prediction).pow(2)

    def get_noise(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return prediction

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return alpha * in_data - sigma * prediction


class VParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        v_target = alpha * noise - sigma * target
        v_weight = (torch.exp(-gamma) + 1)
        return (v_target - prediction).pow(2) / v_weight

    def get_noise(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return sigma * in_data + alpha * prediction

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return alpha * in_data - sigma * prediction


class DataParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        data_weight = torch.exp(gamma)
        return data_weight * (target - prediction).pow(2)

    def get_noise(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return ( in_data-alpha*prediction ) / sigma

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return prediction


class FlowOTParam(torch.nn.Module):
    def invert_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        # Assuming FlowOTScheduler for sampling
        # Eq. 65 Kingma and Gao, 2023
        return torch.sigmoid(-gamma / 2)

    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        # Eq. 77, Kingma and Gao, 2023
        o_target = target - noise
        o_weight = (1 + torch.exp(-gamma * 0.5)).pow(-2)
        return o_weight * (o_target - prediction).pow(2)

    def get_noise(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return in_data - (1-self.get_time(gamma)) * prediction

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return in_data + self.get_time(gamma) * prediction


class EDMParam(torch.nn.Module):
    def __init__(self, sigma_data: float = 1.):
        super().__init__()
        self.sigma_data = sigma_data

    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        # Eq. 119 in Kingma & Gao 2023
        target_factor = (
            torch.sqrt(torch.exp(-gamma) + self.sigma_data**2)
        )/(
            torch.exp(-0.5 * gamma) * self.sigma_data
        )
        in_data_factor = (
            self.sigma_data * alpha
        ) / (
            torch.exp(-0.5 * gamma)
            * torch.sqrt(torch.exp(-gamma) + self.sigma_data**2)
        )
        f_target = target_factor * target - in_data_factor * in_data
        # Eq. 125
        f_weighting = torch.exp(-gamma) / self.sigma_data**2 + 1
        return (f_target-prediction).pow(2)/f_weighting

    def get_noise(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        state = self.forward(
            prediction,
            in_data,
            alpha,
            sigma,
            gamma,
            *args, ** kwargs
        )
        return (in_data - alpha * state) / sigma

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        # Eq. 118
        sigma_tilde = torch.exp(-0.5 * gamma)
        denom = sigma_tilde ** 2 + self.sigma_data ** 2
        in_data_factor = (self.sigma_data ** 2 * alpha) / denom
        f_factor = sigma_tilde * self.sigma_data / denom.sqrt()
        return in_data_factor * in_data + f_factor * prediction
