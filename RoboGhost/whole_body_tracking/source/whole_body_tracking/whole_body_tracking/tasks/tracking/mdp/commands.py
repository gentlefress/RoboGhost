
from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class MotionLoader_Replay:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        motion_file = motion_file
        print("loading motion file:", motion_file)
        import joblib
        self.data = joblib.load(motion_file)
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        time_step_total = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        for i in range(len(self.data)):
            joint_pos_list.append(self.data[i]['joint_pos'])
            joint_vel_list.append(self.data[i]['joint_vel'])
            body_pos_w_list.append(self.data[i]['body_pos_w'])
            body_quat_w_list.append(self.data[i]['body_quat_w'])
            body_lin_vel_w_list.append(self.data[i]['body_lin_vel_w'])
            body_ang_vel_w_list.append(self.data[i]['body_ang_vel_w'])
            time_step_total.append(self.data[i]['joint_pos'].shape[0])
        self.num_motions = len(self.data)
        print(f"Total motions loaded: {self.num_motions}")
        self.fps = self.data[0]["fps"]
        joint_pos_array = np.array(joint_pos_list)  
        joint_vel_array = np.array(joint_vel_list)
        body_pos_w_array = np.array(body_pos_w_list)
        body_quat_w_array = np.array(body_quat_w_list)
        body_lin_vel_w_array = np.array(body_lin_vel_w_list)
        body_ang_vel_w_array = np.array(body_ang_vel_w_list)
        time_step_total_array = np.array(time_step_total)
        self.joint_pos = torch.from_numpy(joint_pos_array).to(dtype=torch.float32, device=device)
        self.joint_vel = torch.from_numpy(joint_vel_array).to(dtype=torch.float32, device=device)
        self._body_pos_w = torch.from_numpy(body_pos_w_array).to(dtype=torch.float32, device=device)
        self._body_quat_w = torch.from_numpy(body_quat_w_array).to(dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.from_numpy(body_lin_vel_w_array).to(dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.from_numpy(body_ang_vel_w_array).to(dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = torch.from_numpy(time_step_total_array).to(dtype=torch.long, device=device)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[ : , : , self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]

    def save_pkl(self,motion_id):
        return self.data[motion_id]

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        motion_file = motion_file
        print("loading motion file:", motion_file)
        import joblib
        data = joblib.load(motion_file)
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        time_step_total = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        for i in range(len(data)):
            joint_pos_list.append(data[i]['joint_pos'])
            joint_vel_list.append(data[i]['joint_vel'])
            body_pos_w_list.append(data[i]['body_pos_w'])
            body_quat_w_list.append(data[i]['body_quat_w'])
            body_lin_vel_w_list.append(data[i]['body_lin_vel_w'])
            body_ang_vel_w_list.append(data[i]['body_ang_vel_w'])
            time_step_total.append(data[i]['joint_pos'].shape[0])
        self.num_motions = len(data)
        print(f"Total motions loaded: {self.num_motions}")
        self.fps = data[0]["fps"]
        self.time_step_total = torch.tensor([len(x) for x in joint_pos_list], 
                                            dtype=torch.long, device=device)
        max_len = self.time_step_total.max().item()
        
        def pad_and_convert(seq_list, max_len):
            feat_shape = seq_list[0].shape[1:]  

            padded_shape = (len(seq_list), max_len) + feat_shape
            padded = np.zeros(padded_shape, dtype=np.float32)
            
            for i, seq in enumerate(seq_list):
                length = min(len(seq), max_len)
                padded[i, :length] = seq[:length]
            return torch.from_numpy(padded).to(device)
        
        self.joint_pos = pad_and_convert(joint_pos_list, max_len)
        self.joint_vel = pad_and_convert(joint_vel_list, max_len)
        self._body_pos_w = pad_and_convert(body_pos_w_list, max_len)
        self._body_quat_w = pad_and_convert(body_quat_w_list, max_len)
        self._body_lin_vel_w = pad_and_convert(body_lin_vel_w_list, max_len)
        self._body_ang_vel_w = pad_and_convert(body_ang_vel_w_list, max_len)
        
        self.mask = torch.arange(max_len, device=device)[None, :] < self.time_step_total[:, None]

        time_step_total_array = np.array(time_step_total)

        self._body_indexes = body_indexes
        self.time_step_total = torch.from_numpy(time_step_total_array).to(dtype=torch.long, device=device)
        latent_path = '/data/lizhe/roboghost/general.pt'
        motion_latent = torch.load(latent_path)
        motion_latent_list = []
        for i in range(len(data)):
            motion_name_array = data[i]['motion_name']
            
            if isinstance(motion_name_array, np.ndarray):
                if motion_name_array.size > 0:
                    motion_name = str(motion_name_array[0])
                else:
                    motion_name_list.append(None)
            else:
                motion_name = str(motion_name_array)
            
            if motion_name in motion_latent:
                latent_vector = motion_latent[motion_name]
                motion_latent_list.append(latent_vector)
            else:
                print(f"lost: {motion_name}")
                motion_latent_list.append(None)
        self.motion_latents = pad_and_convert(motion_latent_list, max_len)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[ : , : , self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_ids = torch.remainder(self.motion_ids, self.motion.num_motions)


        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        # ================== roboghost adaptive sampling ================== #
        self.num_segments = 20 
        self.sampling_para = torch.ones(self.motion.num_motions, self.num_segments, 
                                    device=self.device) / self.num_segments

        self.decay_factor = 0.8  
        self.base_increment = 0.005  
        self.causal_horizon = 4  

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def music(self) -> torch.Tensor:
        return self.motion.music[self.motion_ids, self.time_steps]
    
    @property
    def motion_latent(self) -> torch.Tensor:
        return self.motion.motion_latents[self.motion_ids, self.time_steps]

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)


    def _adaptive_sampling_causal(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        episode_failed = self._env.termination_manager.terminated[env_ids_tensor]
        
        if torch.any(episode_failed):
            failed_env_ids = env_ids_tensor[episode_failed]
            failed_motion_ids = self.motion_ids[failed_env_ids]
            failed_time_steps = self.time_steps[failed_env_ids]
            
            motion_lengths = self.motion.time_step_total[failed_motion_ids]
  
            motion_part = motion_lengths / self.num_segments
            
            segment_pos = (failed_time_steps.float() / motion_part).clamp(0, self.num_segments - 1).long()
            
            para_update = torch.zeros_like(self.sampling_para)
            
            for i in range(len(failed_env_ids)):
                motion_id = failed_motion_ids[i].item()
                seg = segment_pos[i].item()
                
                for offset in range(self.causal_horizon):
                    target_seg = seg - offset
                    if target_seg >= 0:
                        decay = self.decay_factor ** offset  
                        para_update[motion_id, target_seg] += self.base_increment * decay
            
            self.sampling_para += para_update
            
            self.sampling_para = torch.clamp(self.sampling_para, min=1e-6)
            row_sums = self.sampling_para.sum(dim=1, keepdim=True)
            self.sampling_para = self.sampling_para / row_sums

        sampled_segments = []
        for env_id in env_ids_tensor:
            motion_id = self.motion_ids[env_id].item()
            segment = torch.multinomial(self.sampling_para[motion_id], 1).item()
            sampled_segments.append(segment)
        
        sampled_segments_tensor = torch.tensor(sampled_segments, device=self.device, dtype=torch.long)
        
        motion_ids_env = self.motion_ids[env_ids_tensor]
        motion_lengths_env = self.motion.time_step_total[motion_ids_env]
        motion_part_env = motion_lengths_env / self.num_segments
        
        segment_start = sampled_segments_tensor * motion_part_env
        segment_end = (sampled_segments_tensor + 1) * motion_part_env
        
        phase = torch.rand(len(env_ids_tensor), device=self.device)  
        sampled_times = segment_start + phase * (segment_end - segment_start)
        
        sampled_times = torch.clamp(sampled_times, max=motion_lengths_env - 1)
        
        self.time_steps[env_ids_tensor] = sampled_times.long()



    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count[0]) // max(self.motion.time_step_total[0], 1), 0, self.bin_count[0] - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins.long(), minlength=self.bin_count[0])

        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / self.bin_count[0]
        
        sampling_probabilities = sampling_probabilities.unsqueeze(1)  # [num_motions, 1, bin_count]
        padding = self.cfg.adaptive_kernel_size - 1
        padded = torch.nn.functional.pad(
            sampling_probabilities, 
            (padding // 2, padding - padding // 2), 
            mode="replicate"
        )
        kernel = self.kernel.view(1, 1, -1).repeat(self.motion.num_motions, 1, 1)
        smoothed = torch.nn.functional.conv1d(padded, kernel, groups=self.motion.num_motions)
        sampling_probabilities = smoothed.squeeze(1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum(dim=1, keepdim=True)

        env_motion_ids = self.motion_ids[env_ids]
        env_probabilities = sampling_probabilities[env_motion_ids]
        
        sampled_bins = torch.multinomial(env_probabilities, num_samples=1, replacement=True).squeeze(1)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count[0]
            * (self.motion.time_step_total[0] - 1)
        ).long()

    def _random_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        motion_ids = self.motion_ids[env_ids_tensor]
        
        if hasattr(self.motion, 'mask'):
            valid_counts = self.motion.mask[motion_ids].sum(dim=1)
            
            valid_env_mask = valid_counts > 0
            valid_env_ids = env_ids_tensor[valid_env_mask]
            valid_motion_ids = motion_ids[valid_env_mask]
            
            if len(valid_env_ids) > 0:
                random_indices = torch.rand(len(valid_env_ids), device=self.device)
                scaled_indices = (random_indices * valid_counts[valid_env_mask]).long()
                
                valid_time_steps = self.motion.mask[valid_motion_ids].nonzero(as_tuple=True)

                cumsum_counts = torch.cat([torch.tensor([0], device=self.device), valid_counts[valid_env_mask].cumsum(dim=0)[:-1]])
                selected_indices = scaled_indices + cumsum_counts
                
                self.time_steps[valid_env_ids] = valid_time_steps[1][selected_indices]
            
            invalid_env_mask = ~valid_env_mask
            invalid_env_ids = env_ids_tensor[invalid_env_mask]
            invalid_motion_ids = motion_ids[invalid_env_mask]
            
            if len(invalid_env_ids) > 0:
                max_times = self.motion.time_step_total[invalid_motion_ids]
                self.time_steps[invalid_env_ids] = torch.randint(0, max_times.max().item(), 
                                                                 (len(invalid_env_ids),), device=self.device)
                self.time_steps[invalid_env_ids] = torch.min(
                    self.time_steps[invalid_env_ids], 
                    max_times - 1
                )
        else:
            motion_ids = self.motion_ids[env_ids_tensor]
            max_times = self.motion.time_step_total[motion_ids]
            self.time_steps[env_ids_tensor] = torch.randint(0, max_times.max().item(), 
                                                           (len(env_ids_tensor),), device=self.device)
            self.time_steps[env_ids_tensor] = torch.min(
                self.time_steps[env_ids_tensor], 
                max_times - 1
            )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        self._adaptive_sampling_causal(env_ids)
        # self._random_sampling(env_ids)
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=self.device)
  
        root_pos[env_ids_tensor] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids_tensor] = quat_mul(orientations_delta, root_ori[env_ids_tensor])
        
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=self.device)
        root_lin_vel[env_ids_tensor] += rand_samples[:, :3]
        root_ang_vel[env_ids_tensor] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_noise = sample_uniform(*self.cfg.joint_position_range, joint_pos[env_ids_tensor].shape, joint_pos.device)
        joint_pos[env_ids_tensor] += joint_noise
        
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids_tensor]
        joint_pos[env_ids_tensor] = torch.clamp(
            joint_pos[env_ids_tensor], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        
        self.robot.write_joint_state_to_sim(joint_pos[env_ids_tensor], joint_vel[env_ids_tensor], env_ids=env_ids_tensor)
        
        root_states = torch.cat([
            root_pos[env_ids_tensor], 
            root_ori[env_ids_tensor], 
            root_lin_vel[env_ids_tensor], 
            root_ang_vel[env_ids_tensor]
        ], dim=-1)
        self.robot.write_root_state_to_sim(root_states, env_ids=env_ids_tensor)

    def _update_command(self):
        self.time_steps += 1

        motion_ids = self.motion_ids
        max_times = self.motion.time_step_total[motion_ids]
        need_resample = self.time_steps >= max_times
        
        resample_env_ids = need_resample.nonzero(as_tuple=True)[0].tolist()
        
        if resample_env_ids:
            self._resample_command(resample_env_ids)
        
        anchor_pos_w_repeat = self.anchor_pos_w.unsqueeze(1).repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w.unsqueeze(1).repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w.unsqueeze(1).repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w.unsqueeze(1).repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0., 0., 0.)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0., 0., 0.)
