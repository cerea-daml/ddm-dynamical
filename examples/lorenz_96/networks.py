#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27/08/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import torch.nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

import numpy as np

# Internal modules
from ddm_dynamical.layers import SinusoidalEmbedding

main_logger = logging.getLogger(__name__)


class FilmLayer(torch.nn.Module):
    def __init__(
            self,
            n_neurons: int,
            n_conditional: int,
    ):
        super().__init__()
        self.affine_film = torch.nn.Linear(
            n_conditional, n_neurons*2
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedded_time: torch.Tensor
    ) -> torch.Tensor:
        scale_shift = self.affine_film(embedded_time)
        scale_shift = scale_shift.view(
            scale_shift.shape+(1, )*(in_tensor.dim()-scale_shift.dim())
        )
        scale, shift = scale_shift.chunk(2, dim=1)
        filmed_tensor = in_tensor * (scale + 1) + shift
        return filmed_tensor


class L96Padding(torch.nn.Module):
    def __init__(self, pad: _size_2_t = 1):
        super().__init__()
        self.pad = _pair(pad)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        padded_tensor = F.pad(in_tensor, self.pad, mode="circular")
        return padded_tensor


class L96Conv(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
    ):
        super().__init__()
        self.conv_layer = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=0
        )
        self.padding = L96Padding((kernel_size-1) // 2)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        padded_tensor = self.padding(in_tensor)
        out_tensor = self.conv_layer(padded_tensor)
        return out_tensor


class ConditionedNorm(torch.nn.Module):
    def __init__(self, n_neurons: int, n_embedding: int = 128):
        super().__init__()
        self.norm = torch.nn.GroupNorm(1, n_neurons, eps=1E-6, affine=False)
        self.film = FilmLayer(n_neurons, n_embedding)

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedded_time: torch.Tensor
    ) -> torch.Tensor:
        norm_tensor = self.norm(in_tensor)
        return self.film(norm_tensor, embedded_time)


class ConvNextBlock(torch.nn.Module):
    def __init__(
            self,
            n_channels: int,
            kernel_size: int = 3,
            n_embedding: int = 128,
            mixing_mult: int = 1,
            layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.spatial_extraction = L96Conv(
            n_channels, n_channels, kernel_size=kernel_size,
            stride=1, groups=n_channels
        )
        self.norm = ConditionedNorm(n_channels, n_embedding)
        self.mixing_layers = torch.nn.Sequential(
            L96Conv(n_channels, n_channels * mixing_mult, 1),
            torch.nn.ReLU(),
            L96Conv(n_channels * mixing_mult, n_channels, 1),
        )
        self.gamma = torch.nn.Parameter(
            torch.full((n_channels, 1), layer_scale_init_value),
            requires_grad=True
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        branch_output = self.spatial_extraction(in_tensor)
        branch_output = self.norm(branch_output, time_tensor)
        branch_output = self.mixing_layers(branch_output)
        return in_tensor + branch_output * self.gamma


class DownLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            kernel_size: int = 3,
            mixing_mult: int = 2,
            n_embedding: int = 128,
    ):
        super().__init__()
        self.norm_layer = ConditionedNorm(in_channels, n_embedding)
        self.pooling = L96Conv(
            in_channels, out_channels, kernel_size=3, stride=2
        )
        out_layers = [
            ConvNextBlock(
                n_channels=out_channels, kernel_size=kernel_size,
                mixing_mult=mixing_mult, n_embedding=n_embedding
            )
        ]
        for block in range(1, n_blocks):
            out_layers.append(
                ConvNextBlock(
                    n_channels=out_channels, kernel_size=kernel_size,
                    mixing_mult=mixing_mult, n_embedding=n_embedding
                )
            )
        self.out_layers = torch.nn.ModuleList(out_layers)

    def forward(
            self,
            in_tensor: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        normed_tensor = self.norm_layer(in_tensor, time_tensor)
        out_tensor = self.pooling(normed_tensor)
        for layer in self.out_layers:
            out_tensor = layer(out_tensor, time_tensor)
        return out_tensor


class UpLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            kernel_size: int = 3,
            mixing_mult: int = 2,
            n_embedding: int = 128,
    ):
        super().__init__()
        self.norm_layer = ConditionedNorm(in_channels, n_embedding)
        self.upscaling = torch.nn.Upsample(
                scale_factor=2, mode='nearest'
        )
        self.up_conv = L96Conv(
            in_channels + out_channels, out_channels, kernel_size=3
        )
        out_layers = [
            ConvNextBlock(
                n_channels=out_channels, kernel_size=kernel_size,
                mixing_mult=mixing_mult, n_embedding=n_embedding
            )
        ]
        for block in range(1, n_blocks):
            out_layers.append(
                ConvNextBlock(
                    n_channels=out_channels, kernel_size=kernel_size,
                    mixing_mult=mixing_mult, n_embedding=n_embedding
                )
            )
        self.out_layers = torch.nn.ModuleList(out_layers)

    def forward(
            self,
            in_tensor: torch.Tensor,
            shortcut: torch.Tensor,
            time_tensor: torch.Tensor
    ) -> torch.Tensor:
        normed_tensor = self.norm_layer(in_tensor, time_tensor)
        upscaled_tensor = self.upscaling(normed_tensor)
        out_tensor = torch.cat([upscaled_tensor, shortcut], dim=1)
        out_tensor = self.up_conv(out_tensor)
        for layer in self.out_layers:
            out_tensor = layer(out_tensor, time_tensor)
        return out_tensor


class UNeXt(torch.nn.Module):
    def __init__(
            self,
            n_channels: int = 64,
            n_blocks: int = 1,
            n_depth: int = 1,
            kernel_size: int = 5,
            n_embedding: int = 512
    ):
        super().__init__()
        self.gamma_embedding = SinusoidalEmbedding(n_embedding)
        self.init_layer = L96Conv(1, n_channels, kernel_size=7)

        self.init_blocks = torch.nn.ModuleList([
            ConvNextBlock(
                n_channels=n_channels,
                kernel_size=kernel_size,
                n_embedding=n_embedding
            )
            for _ in range(n_blocks)
        ])
        self.down_layers, self.up_layers = self._const_net(
            n_features=n_channels,
            n_depth=n_depth,
            n_blocks=n_blocks,
            n_embedding=n_embedding,
            mixing_mult=1,
            kernel_size=kernel_size
        )
        bottleneck_features = n_channels * (2**n_depth)
        self.bottleneck_layer = ConvNextBlock(
            bottleneck_features,
            kernel_size=kernel_size,
            n_embedding=n_embedding
        )
        self.out_blocks = torch.nn.ModuleList([
            ConvNextBlock(
                n_channels=n_channels,
                kernel_size=kernel_size,
                n_embedding=n_embedding
            )
            for _ in range(n_blocks)
        ])
        self.output_head = L96Conv(n_channels, 1)

    @staticmethod
    def _const_net(
            n_features: int = 64,
            n_depth: int = 3,
            n_blocks: int = 2,
            n_embedding: int = 128,
            mixing_mult: int = 2,
            kernel_size: int = 5,
    ) -> Tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        down_layers = []
        n_down_feature_list = n_features * (2 ** np.arange(n_depth + 1))
        for k, out_features in enumerate(n_down_feature_list[1:]):
            in_features = n_down_feature_list[k]
            down_layers.append(
                DownLayer(
                    in_features,
                    out_features,
                    n_blocks,
                    kernel_size=kernel_size,
                    mixing_mult=mixing_mult,
                    n_embedding=n_embedding
                )
            )
        up_layers = []
        n_up_feature_list = n_down_feature_list[::-1]
        for k, out_features in enumerate(n_up_feature_list[1:]):
            in_features = n_up_feature_list[k]
            up_layers.append(
                UpLayer(
                    in_features,
                    out_features,
                    n_blocks,
                    kernel_size=kernel_size,
                    mixing_mult=mixing_mult,
                    n_embedding=n_embedding
                )
            )
        down_layers = torch.nn.ModuleList(down_layers)
        up_layers = torch.nn.ModuleList(up_layers)
        return down_layers, up_layers

    def extract_features(
            self,
            in_tensor: torch.Tensor,
            normalized_gamma: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.gamma_embedding(normalized_gamma)
        in_tensor = in_tensor.unsqueeze(-2)
        init_tensor = self.init_layer(in_tensor)
        for layer in self.init_blocks:
            init_tensor = layer(init_tensor, embedding)
        down_tensor_list = [init_tensor]
        for layer in self.down_layers:
            down_tensor_list.append(
                layer(down_tensor_list[-1], embedding)
            )
        down_tensor_list = down_tensor_list[::-1]
        features_tensor = self.bottleneck_layer(
            down_tensor_list[0], embedding
        )
        for k, layer in enumerate(self.up_layers):
            features_tensor = layer(
                features_tensor, down_tensor_list[k+1], embedding
            )
        for layer in self.out_blocks:
            features_tensor = layer(features_tensor, embedding)
        return features_tensor

    def forward(self, in_tensor, normalized_gamma, mask=None, **kwargs):
        extracted_features = self.extract_features(
            in_tensor=in_tensor, normalized_gamma=normalized_gamma
        )
        out_tensor = self.output_head(extracted_features)
        out_tensor = out_tensor.squeeze(dim=-2)
        return out_tensor
