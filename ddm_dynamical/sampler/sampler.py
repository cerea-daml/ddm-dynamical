#!/bin/env python
# -*- coding: utf-8 -*-
#
#
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Callable, Dict, Any

# External modules
import torch
from tqdm.autonotebook import tqdm

# Internal modules
from .defaults import *
from ddm_dynamical.parameterization import EPSParam
from ddm_dynamical.utils import normalize_gamma


logger = logging.getLogger(__name__)


class BaseSampler(torch.nn.Module):
    def __init__(
            self,
            scheduler: "ddm_dynamical.scheduler.noise_scheduler.NoiseScheduler",
            timesteps: int = 250,
            denoising_network: torch.nn.Module = None,
            pre_func: Callable = None,
            post_func: Callable = None,
            prior_sampler: Callable = None,
            param: Callable = None,
            gamma_min: float = -15.,
            gamma_max: float = 15.,
            pbar: bool = True,
            sample_kwargs: Dict[str, Any] = None
    ):
        super().__init__()
        self.denoising_network = denoising_network
        self.pre_func = pre_func or default_preprocessing
        self.post_func = post_func or default_postprocessing
        self.prior_sampler = prior_sampler or default_prior_sample
        self.param = param or EPSParam()
        self.timesteps = timesteps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.scheduler = scheduler
        self.sample_kwargs = sample_kwargs
        self.pbar = pbar

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        if self._sample_kwargs is None:
            try:
                template = next(self.denoising_network.parameters())
                self._sample_kwargs = {
                    "device": template.device,
                    "layout": template.layout,
                    "dtype": template.dtype
                }
            except (StopIteration, AttributeError):
                self._sample_kwargs = dict()
        return self._sample_kwargs

    @sample_kwargs.setter
    def sample_kwargs(self, kwargs: Dict[str, Any] = None) -> None:
        self._sample_kwargs = kwargs

    def convert_step(self, step: torch.Tensor):
        gamma = self.scheduler(step)
        alpha_sq = torch.sigmoid(gamma)
        alpha = alpha_sq.sqrt()
        sigma = (1-alpha_sq).sqrt()
        return {
            "step": step, "gamma": gamma, "alpha_sq": alpha_sq,
            "alpha": alpha, "sigma": sigma
        }

    def estimate_prediction(
            self,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        in_tensor = self.pre_func(
            in_data=in_data,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            **conditioning
        )
        norm_gamma = torch.ones(
            in_tensor.size(0), 1, device=in_tensor.device, dtype=in_tensor.dtype
        ) * normalize_gamma(gamma, self.gamma_min, self.gamma_max)
        prediction = self.denoising_network(
            in_tensor, normalized_gamma=norm_gamma, **conditioning
        )
        prediction = self.post_func(
            prediction=prediction,
            in_data=in_data,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            **conditioning
        )
        return prediction

    def estimate_denoised(
            self,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        prediction = self.estimate_prediction(
            in_data=in_data,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            **conditioning
        )
        denoised = self.param(
            prediction=prediction,
            in_data=in_data,
            alpha=alpha,
            sigma=sigma,
            gamma=gamma,
            **conditioning
        )
        return denoised



    @torch.no_grad()
    def sample(
            self,
            sample_shape=torch.Size([]),
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        prior_sample = self.prior_sampler(sample_shape, **self.sample_kwargs)
        denoised_data = self.reconstruct(
            prior_sample, self.timesteps, **conditioning
        )
        return denoised_data

    def reconstruct(
            self,
            in_tensor: torch.Tensor,
            n_steps: int = 250,
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        denoised_tensor = in_tensor
        time_steps = torch.linspace(
            1, 0, self.timesteps+1,
            device=in_tensor.device, dtype=in_tensor.dtype,
            layout=in_tensor.layout
        )[-n_steps-1:]
        pbar = enumerate(time_steps[1:])
        if self.pbar:
            pbar = tqdm(pbar, total=n_steps, leave=False)
        for k, next_step in pbar:
            curr_stats = self.convert_step(time_steps[k])
            next_stats = self.convert_step(next_step)
            denoised_tensor = self(
                denoised_tensor, curr_stats, next_stats, **conditioning
            )
        return denoised_tensor

    def forward(
            self,
            in_data: torch.Tensor,
            curr_stats: Dict[str, torch.Tensor],
            next_stats: Dict[str, torch.Tensor],
            **conditioning: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pass
