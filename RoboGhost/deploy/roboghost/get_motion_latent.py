import pickle
import torch
import numpy as np
import torch.nn.functional as F

class get_motion_latent():
    def __init__(self):
        self._device = 'cpu'
        self.load_motion_latent()
        self.pad_and_reorganize()

    def load_motion_latent(self):
        motion_latent_path = '/home/baai/humanoid/roboghost/unitree_rl_gym/roboghost/motion_latent/general.pt'
        self.motion_latents = torch.load(motion_latent_path)

    def pad_and_reorganize(self):

        self.sorted_keys = sorted(self.motion_latents.keys())
        
        self.max_frames = 0
        for key in self.sorted_keys:
            num_frames = self.motion_latents[key].shape[0]
            self.max_frames = max(self.max_frames, num_frames)
        
        print(f"Max frames: {self.max_frames}")
        
        sample_key = self.sorted_keys[0]
        latent_dim = self.motion_latents[sample_key].shape[1]
        
        num_motions = len(self.sorted_keys)
        self.motion_latents_tensor = torch.zeros(
            (num_motions, self.max_frames, latent_dim), 
            device=self._device
        )
        
        self.valid_frames_mask = torch.zeros(
            (num_motions, self.max_frames), 
            dtype=torch.bool,
            device=self._device
        )
        
        for i, key in enumerate(self.sorted_keys):
            tensor = self.motion_latents[key]
            num_frames = tensor.shape[0]
            
            self.motion_latents_tensor[i, :num_frames, :] = tensor
            
            self.valid_frames_mask[i, :num_frames] = True
        
        self.original_lengths = [
            self.motion_latents[key].shape[0] 
            for key in self.sorted_keys
        ]

    def get_motion_latent(self, motion_ids, time_steps) -> torch.Tensor:
        
        if isinstance(motion_ids, torch.Tensor):
            motion_id = motion_ids.item() if motion_ids.numel() == 1 else int(motion_ids.cpu().numpy()[0])
        elif isinstance(motion_ids, (list, np.ndarray)):
            motion_id = int(motion_ids[0])
        else:
            motion_id = int(motion_ids)
        

        if isinstance(time_steps, torch.Tensor):
            timestep = time_steps.item() if time_steps.numel() == 1 else int(time_steps.cpu().numpy()[0])
        elif isinstance(time_steps, (list, np.ndarray)):
            timestep = int(time_steps[0])
        else:
            timestep = int(time_steps)
        

        key = self.sorted_keys[motion_id]
        original_length = self.original_lengths[motion_id]
        
        if timestep >= original_length:
            timestep = timestep % original_length
        return self.motion_latents[key][timestep]