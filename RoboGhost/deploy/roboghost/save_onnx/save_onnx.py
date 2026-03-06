#!/usr/bin/env python3
"""
ONNX conversion script for g1_stu_future (student policy with future motion support)
Usage: python save_onnx_stu_future.py --ckpt_path <absolute_path_to_checkpoint>
"""

import os, sys
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from diffusion_policy import DiffusionPolicy
import argparse
from termcolor import cprint
import numpy as np

class EmpiricalNormalization(nn.Module):

    def __init__(self, shape, eps=1e-2, until=None):

        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)
    
    @torch.jit.unused
    def update(self, x):

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

class HardwareStudentNN(nn.Module):
    """Hardware deployment wrapper for student policy with future motion support."""
    
    def __init__(self, num_actor_obs, num_actions, device):
        super().__init__()
        self.student_distill_net = DiffusionPolicy(num_actor_obs, num_actions)
        self.student_obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(device)
    def forward(self, obs):
        return self.student_distill_net.sample(self.student_obs_normalizer(obs))

def convert_to_onnx(args):
    """Convert g1_stu policy to ONNX."""

    ckpt_path = args.ckpt_path
    
    # Check if checkpoint file exists
    if not os.path.exists(ckpt_path):
        cprint(f"Error: Checkpoint file not found: {ckpt_path}", "red")
        return
    
    # G1 student future configuration - EXACT DIMENSIONS FROM DEBUG
    robot_name = "g1"
    num_actions = 23
    history_len = 15
    motion_latent_dims = 64
    obs_prop_dims = 72
    num_observations = motion_latent_dims + obs_prop_dims * history_len
    
    # Network architecture parameters
    
    print(f"G1 Student Future Policy Configuration:")
    print(f"  Robot: {robot_name}")
    print(f"  Actions: {num_actions}")
    
    device = torch.device('cpu')
    policy = HardwareStudentNN(num_observations, num_actions, device).to(device)

    # Load trained model
    cprint(f"Loading model from: {ckpt_path}", "green")
    
    # Load with weights_only=False to avoid the warning for now
    ac_state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    from collections import OrderedDict
    latent_encoder = OrderedDict()
    student_actor_backbone = OrderedDict()
    for k, v in ac_state_dict['student_model_state_dict'].items():
        if k.startswith('latent_encoder'):
            new_key = k.replace('latent_encoder.', '')
            latent_encoder[new_key] = v
        
        if k.startswith('student_actor_backbone'):
            new_key = k.replace('student_actor_backbone.', '')
            student_actor_backbone[new_key] = v
    ckpt_norm = OrderedDict()
    for k, v in ac_state_dict['student_obs_norm_state_dict'].items():
        if k.startswith('student_obs_norm_state_dict'):
            new_key = k.replace('student_obs_norm_state_dict.', '')
        else:
            new_key = k  
        ckpt_norm[f'{new_key}'] = v
    policy.student_distill_net.student_actor_backbone.load_state_dict(student_actor_backbone, strict=False)
    policy.student_distill_net.latent_encoder.load_state_dict(latent_encoder, strict=False)
    policy.student_obs_normalizer.load_state_dict(ckpt_norm, strict=False)
    
    policy = policy.to(device)
    
    # Export to ONNX with same name but .onnx extension
    policy.eval()
    with torch.no_grad(): 
        # Create dummy input with correct observation structure
        batch_size = 1  # Use batch size 1 for simplicity
        obs_input = torch.ones(batch_size, num_observations, device=device)
        cprint(f"Input observation shape: {obs_input.shape}", "cyan")
        
        # Generate ONNX path with same name but .onnx extension
        onnx_path = ckpt_path.replace('.pt', '.onnx')
        
        # Export to ONNX
        torch.onnx.export(
            policy,
            obs_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        cprint(f"ONNX model saved to: {onnx_path}", "green")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert g1_stu_future student policy to ONNX')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Absolute path to checkpoint file')
    args = parser.parse_args()
    convert_to_onnx(args)