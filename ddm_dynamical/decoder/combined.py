#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import List, Tuple, Dict

# External modules
import torch.nn

# Internal modules
from .base_decoder import BaseDecoder

main_logger = logging.getLogger(__name__)


class CombinedDecoder(BaseDecoder):
    def __init__(
            self,
            base_decoders: Dict[str, torch.nn.Module],
            stochastic: bool = False,
            **kwargs
    ):
        super().__init__(stochastic=stochastic)
        self.base_decoders = torch.nn.ModuleDict(base_decoders)
        for decoder in self.base_decoders.values():
            decoder.stochastic = stochastic
            for k, v in kwargs.items():
                decoder.__setattr__(k, v)

    @property
    def splits(self) -> List[int]:
        return [d.n_dims for d in self.base_decoders.values()]

    def enumerate_decoder_in(
            self,
            in_tensor: torch.Tensor
    ) -> enumerate[int, Tuple[torch.nn.Module, torch.Tensor]]:
        splitted_tensor = in_tensor.split(self.splits, dim=1)
        decoder_in = zip(self.base_decoders.values(), splitted_tensor)
        return enumerate(decoder_in)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.tensor:
        return torch.cat([
            decoder(
                in_tensor,
                first_guess[:, [k]],
                mask
            )
            for k, (decoder, in_tensor)
            in self.enumerate_decoder_in(in_tensor)
        ], dim=1)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        for k, (decoder, in_tensor)\
                in self.enumerate_decoder_in(in_tensor):
            decoder.update(
                in_tensor,
                first_guess[:, [k]],
                target[:, [k]],
                mask
            )

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = []
        loss_clim = []
        for k, (decoder, in_tensor) in self.enumerate_decoder_in(in_tensor):
            curr_loss, curr_clim = decoder.loss(
                in_tensor,
                first_guess[:, [k]],
                target[:, [k]],
                mask
            )
            loss.append(curr_loss)
            loss_clim.append(curr_clim)
        return torch.mean(torch.stack(loss)), torch.mean(torch.stack(loss_clim))
