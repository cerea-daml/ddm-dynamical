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
from math import inf
from typing import Union

# External modules
import torch
import torch.nn
import torch.nn.functional as F

# Internal modules
from ..utils import masked_average
from .delta import DeltaDecoder

main_logger = logging.getLogger(__name__)


class LowerDeltaDecoder(DeltaDecoder):
    n_dims = 2

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
    ):
        prediction = super()(in_tensor[:, [0]], first_guess, mask)
        unbounded = in_tensor[:, [1]] > 0
        return torch.where(unbounded, prediction, self.bound)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ):
        combined_mask = self.target_to_unbounded(target) * mask
        return super().update(
            in_tensor[:, [0]], first_guess, target, combined_mask
        )

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        unbounded = self.target_to_unbounded(target)

        ## Regression loss for unbounded
        loss_reg = super().loss(
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
        return loss_reg + loss_class


class UpperDeltaDecoder(LowerDeltaDecoder):
    n_dims = 2

    @property
    def bound(self) -> float:
        return self.upper_bound

    def target_to_unbounded(self, target: torch.Tensor) -> torch.Tensor:
        return (target < self.upper_bound).to(target)


class BoundedDeltaDecoder(DeltaDecoder):
    n_dims = 4

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ):
        # Get prediction
        prediction = super()(
            in_tensor[:, [0]], first_guess, mask
        ).squeeze(dim=1)

        # Add cases for lower and upper bound
        prediction_cases = torch.stack([
            torch.full_like(prediction, self.lower_bound),
            torch.full_like(prediction, self.upper_bound),
            prediction
        ], dim=-1)

        # Convert logits to one hot tensor
        case_idx = torch.argmax(in_tensor[:, 1:], dim=1, keepdim=False)
        case_mask = F.one_hot(case_idx)
        return case_mask * prediction_cases

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ):
        unbounded = torch.logical_and(
            target > self.lower_bound, target < self.upper_bound
        )
        combined_mask = unbounded * mask
        return super().update(
            in_tensor[:, [0]], first_guess, target, combined_mask
        )

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        lower_bounded = target <= self.lower_bound
        upper_bounded = target >= self.upper_bound
        unbounded = torch.logical_and(
            ~lower_bounded, ~upper_bounded
        ).to(mask)

        ## Regression loss for unbounded values
        loss_reg = super().loss(
            in_tensor=in_tensor[:, [0]],
            first_guess=first_guess,
            target=target,
            mask=mask*unbounded
        )

        ## Classification loss for bounds
        target_one_hot = torch.stack(
            [lower_bounded, upper_bounded, unbounded], dim=-1
        ).squeeze(dim=1)
        target_idx = torch.argmax(target_one_hot, dim=1)
        loss_class = F.cross_entropy(
            in_tensor[:, 1:], target_idx, reduction="none"
        )
        loss_class = masked_average(loss_class, mask)
        return loss_reg + loss_class
