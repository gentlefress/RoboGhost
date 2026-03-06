# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import inspect
import torch.nn.functional as F
from .DiffMLPs import DiffMLPs_models
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


class DiffusionPolicy(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        **kwargs,
    ):
        if kwargs:
            print(
                "DiffusionPlicy.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.motion_latent_dim = 64
        activation = nn.ELU()
        latent_encoder = []
        latent_encoder.append(nn.Linear(self.motion_latent_dim, 128))
        latent_encoder.append(activation)
        latent_encoder.append(nn.Linear(128, 64))
        latent_encoder.append(activation)
        self.latent_encoder = nn.Sequential(*latent_encoder)
        self.student_actor_backbone = DiffMLPs_models['DDPM-XL'](target_channels=23, z_channels=20)
        print(f"Diffusion Policy: {self.student_actor_backbone}")
    
    def sample(self, x):    
        motion_latent = x[:, :self.motion_latent_dim]
        x1 = x[:, self.motion_latent_dim:]
        motion_latent_encoder = self.latent_encoder(motion_latent)
        new_input = torch.cat([motion_latent_encoder, x1], dim=-1)
        return self.student_actor_backbone.sample(new_input, cfg=4.5)


    def forward(self, action_teacher, x):
        motion_latent = x[:, :self.motion_latent_dim]
        x1 = x[:, self.motion_latent_dim:]
        motion_latent_encoder = self.latent_encoder(motion_latent)
        new_input = torch.cat([motion_latent_encoder, x1], dim=-1)
        backbone_output = self.student_actor_backbone(action_teacher, new_input)
        return backbone_output

