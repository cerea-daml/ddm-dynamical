#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10/11/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import torch
import torch.nn
import torch.nn.functional as F

# Internal modules
from .base_decoder import BaseDecoder
from ..utils import masked_average

main_logger = logging.getLogger(__name__)


class LowerBoundedDecoder(BaseDecoder):
    def __init__(self, regression_decoder: BaseDecoder):
        super().__init__()
        self.regression_decoder = regression_decoder
        self.n_dims = 1 + self.regression_decoder.n_dims

    @property
    def bound(self) -> float:
        return self.lower_bound

    def target_to_unbounded(self, target: torch.Tensor) -> torch.Tensor:
        return (target > self.lower_bound).to(target)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = self.regression_decoder.forward(
            in_tensor[:, [0]], first_guess, mask
        )
        if self.stochastic:
            unbounded = torch.bernoulli(
                torch.sigmoid(in_tensor[:, [1]])
            ).bool()
        else:
            unbounded = in_tensor[:, [1]] > 0
        return torch.where(unbounded, prediction, self.bound)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        combined_mask = self.target_to_unbounded(target) * mask
        return self.regression_decoder.update(
            in_tensor[:, [0]], first_guess, target, combined_mask
        )

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unbounded = self.target_to_unbounded(target)

        ## Regression loss for unbounded
        loss_reg, loss_clim = self.regression_decoder.loss(
            in_tensor=in_tensor[:, [0]],
            first_guess=first_guess,
            target=target,
            mask=mask*unbounded
        )

        ## Classification loss for bounded
        loss_class = F.binary_cross_entropy_with_logits(
            in_tensor[:, [1]], unbounded, reduction="none"
        )
        loss_class = masked_average(loss_class, mask)
        return loss_reg + loss_class, loss_clim + loss_class


class UpperBoundedDecoder(LowerBoundedDecoder):
    @property
    def bound(self) -> float:
        return self.upper_bound

    def target_to_unbounded(self, target: torch.Tensor) -> torch.Tensor:
        return (target < self.upper_bound).to(target)


class BoundedDecoder(BaseDecoder):
    def __init__(self, regression_decoder: BaseDecoder):
        super().__init__()
        self.regression_decoder = regression_decoder
        self.n_dims = 3 + self.regression_decoder.n_dims

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        # Get prediction
        prediction = self.regression_decoder.forward(
            in_tensor[:, [0]], first_guess, mask
        ).squeeze(dim=1)

        # Add cases for lower and upper bound
        prediction_cases = torch.stack([
            torch.full_like(prediction, self.lower_bound),
            torch.full_like(prediction, self.upper_bound),
            prediction
        ], dim=-1)

        # Convert logits to one hot tensor
        logits = in_tensor[:, 1:]
        if self.stochastic:
            gumbel_sample = -torch.log(-torch.log(torch.rand_like(logits)))
            logits.add_(gumbel_sample)
        case_idx = torch.argmax(logits, dim=1, keepdim=False)
        case_mask = F.one_hot(case_idx, num_classes=3)
        return (case_mask * prediction_cases).sum(dim=-1)[:, None]

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        unbounded = torch.logical_and(
            target > self.lower_bound, target < self.upper_bound
        )
        combined_mask = unbounded * mask
        return self.regression_decoder.update(
            in_tensor[:, [0]], first_guess, target, combined_mask
        )

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_bounded = target <= self.lower_bound
        upper_bounded = target >= self.upper_bound
        unbounded = torch.logical_and(
            ~lower_bounded, ~upper_bounded
        ).to(mask)

        ## Regression loss for unbounded values
        loss_reg, loss_clim = self.regression_decoder.loss(
            in_tensor=in_tensor[:, [0]],
            first_guess=first_guess,
            target=target,
            mask=mask*unbounded
        )

        ## Classification loss for bounds
        target_one_hot = torch.stack(
            [lower_bounded, upper_bounded, unbounded], dim=-1
        ).squeeze(dim=1)
        target_idx = torch.argmax(target_one_hot, dim=-1)
        loss_class = F.cross_entropy(
            in_tensor[:, 1:], target_idx, reduction="none"
        )
        loss_class = masked_average(loss_class[:, None], mask)
        return loss_reg + loss_class, loss_clim + loss_class
