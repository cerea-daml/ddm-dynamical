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
from typing import List, Tuple

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class CombinedDecoder(torch.nn.Module):
    def __init__(self, base_decoders: List[torch.nn.Module]):
        super().__init__()
        self.base_decoders = base_decoders

    @property
    def splits(self) -> List[int, ...]:
        return [d.n_dims for d in self.base_decoders]

    def enumerate_decoder_in(
            self,
            in_tensor: torch.Tensor
    ) -> enumerate[int, Tuple[torch.nn.Module, torch.Tensor]]:
        splitted_tensor = in_tensor.split(self.splits, dim=1)
        decoder_in = zip(self.base_decoders, splitted_tensor)
        return enumerate(decoder_in)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ):
        return torch.cat([
            decoder(in_tensor, first_guess[:, k], mask)
            for k, (decoder, in_tensor)
            in self.enumerate_decoder_in(in_tensor)
        ], dim=1)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ):
        for k, (decoder, in_tensor)\
                in self.enumerate_decoder_in(in_tensor):
            decoder.update(
                in_tensor, first_guess[:, [k]], target[:, [k]], mask
            )

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(
            torch.stack([
                decoder.loss(
                    in_tensor, first_guess[:, [k]], target[:, [k]], mask
                )
                for k, (decoder, in_tensor)
                in self.enumerate_decoder_in(in_tensor)
            ])
        )
