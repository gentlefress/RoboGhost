import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml


import numpy as np
import mujoco


import onnxruntime

import signal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_motion_latent import get_motion_latent

xml_path = "./unitree_description/mjcf/g1_liao.xml"

# Total simulation time
simulation_duration = 300.0
# Simulation time step
simulation_dt = 0.002
# Controller update frequency (meets the requirement of simulation_dt * controll_decimation=0.02; 50Hz)
control_decimation = 10
    
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def expand_23d_to_29d(actions_23d):
    batch_size = actions_23d.shape[0]
    device = actions_23d.device
    
    actions_29d = torch.zeros(batch_size, 29, device=device)
    actions_29d[:, :23] = actions_23d[:, :]

    return actions_29d


def init_history_buffers_with_step0(d, m, joint_seq, joint_pos_array_seq, history_len=4):
    
    qpos_xml = d.qpos[7:7 + 29]
    qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
    joint_pos_diff = qpos_seq - joint_pos_array_seq
    
    qvel_xml = d.qvel[6:6 + 29]
    qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
    
    base_ang_vel = d.qvel[3:6]
    base_ang_vel_his = np.tile(base_ang_vel[None, :], (history_len, 1))
    joint_pos_his = np.tile(joint_pos_diff[None, :23], (history_len, 1))
    joint_vel_his = np.tile(qvel_seq[None, :23], (history_len, 1))
    action_his = np.zeros((history_len, 23), dtype=np.float32)  
    
    return {
        'base_ang_vel_his': base_ang_vel_his,
        'joint_pos_his': joint_pos_his,
        'joint_vel_his': joint_vel_his,
        'action_his': action_his
    }

joint_xml = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

joint_seq = ['left_hip_pitch_joint', 
             'right_hip_pitch_joint', 
             'waist_yaw_joint', 
             'left_hip_roll_joint', 
             'right_hip_roll_joint', 
             'waist_roll_joint', 
             'left_hip_yaw_joint', 
             'right_hip_yaw_joint', 
             'waist_pitch_joint', 
             'left_knee_joint', 
             'right_knee_joint', 
             'left_shoulder_pitch_joint', 
             'right_shoulder_pitch_joint', 
             'left_ankle_pitch_joint', 
             'right_ankle_pitch_joint', 
             'left_shoulder_roll_joint', 
             'right_shoulder_roll_joint', 
             'left_ankle_roll_joint', 
             'right_ankle_roll_joint', 
             'left_shoulder_yaw_joint', 
             'right_shoulder_yaw_joint', 
             'left_elbow_joint', 
             'right_elbow_joint', 
             'left_wrist_roll_joint', 
             'right_wrist_roll_joint', 
             'left_wrist_pitch_joint', 
             'right_wrist_pitch_joint', 
             'left_wrist_yaw_joint', 
             'right_wrist_yaw_joint']

import numpy as np

joint_pos_array = np.array([-0.312,  0.   ,  0.   ,  0.669, -0.363,  0.   , -0.312,  0.   ,
        0.   ,  0.669, -0.363,  0.   ,  0.   ,  0.   ,  0.   ,  0.2  ,
        0.2  ,  0.   ,  0.6  ,  0.   ,  0.   ,  0.   ,  0.2  , -0.2  ,
        0.   ,  0.6  ,  0.   ,  0.   ,  0.   ])
joint_pos_array_seq = np.array([-0.312, -0.312,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.669,  0.669,  0.2  ,  0.2  , -0.363, -0.363,  0.2  ,
       -0.2  ,  0.   ,  0.   ,  0.   ,  0.   ,  0.6  ,  0.6  ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ])
stiffness_array = np.array([40.179, 99.098, 40.179, 99.098, 28.501, 28.501, 40.179, 99.098,
       40.179, 99.098, 28.501, 28.501, 40.179, 28.501, 28.501, 14.251,
       14.251, 14.251, 14.251, 14.251, 16.778, 16.778, 14.251, 14.251,
       14.251, 14.251, 14.251, 16.778, 16.778])

damping_array = np.array([2.558, 6.309, 2.558, 6.309, 1.814, 1.814, 2.558, 6.309, 2.558,
       6.309, 1.814, 1.814, 2.558, 1.814, 1.814, 0.907, 0.907, 0.907,
       0.907, 0.907, 1.068, 1.068, 0.907, 0.907, 0.907, 0.907, 0.907,
       1.068, 1.068])
action_scale = np.array([0.548, 0.548, 0.548, 0.351, 0.351, 0.439, 0.548, 0.548, 0.439,
       0.351, 0.351, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
       0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,
       0.075, 0.075])

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    num_actions = 29
    num_obs = 1144
    import onnx
    
    # your checkpoint path
    parser = argparse.ArgumentParser(description="Sim2Sim with ONNX policy.")
    parser.add_argument("--onnx_path", type=str, required=True, help="The path of onnx model.")
    parser.add_argument("--motion_latent_path", type=str, required=True, help="The path of motion latent.")
    parser.add_argument("--motion_id", type=int, required=True, help="select motion id.")
    cli_args = parser.parse_args()
    policy_path = cli_args.onnx_path
    print('load_policy_from:', policy_path)
    model = onnxruntime.InferenceSession(policy_path)
    metadata = {
        'simulation_dt': simulation_dt,
        'xml_path': xml_path,
        'joint_names': joint_xml
    }

    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

                
    policy = model
    load_motion_latent = get_motion_latent(motion_latent_path=cli_args.motion_latent_path)
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0
    motion_id = torch.tensor([cli_args.motion_id], device=load_motion_latent._device)
    print("Current IDs", motion_id.item())
    target_dof_pos = joint_pos_array.copy()
    d.qpos[7:] = target_dof_pos
    body_name = "torso_link"
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")
    history_len = 15
    history_buffers = init_history_buffers_with_step0(d, m, joint_seq, joint_pos_array_seq, history_len)
    with mujoco.viewer.launch_passive(m, d) as viewer:

        start = time.time()
        try:
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()

                mujoco.mj_step(m, d)
                tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)# xml

                d.ctrl[:] = tau
                counter += 1
                if counter % control_decimation == 0:

                    motion_latent = load_motion_latent.get_motion_latent(motion_id, timestep)

                    qpos_xml = d.qpos[7:7 + num_actions]
                    qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                    joint_pos_diff = qpos_seq - joint_pos_array_seq
                    
                    qvel_xml = d.qvel[6:6 + num_actions]
                    qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                    
                    base_ang_vel = d.qvel[3:6]
                    
                    for buffer_name in ['base_ang_vel_his', 'joint_pos_his', 'joint_vel_his']:
                        history_buffers[buffer_name][:-1] = history_buffers[buffer_name][1:]
                    
                    history_buffers['base_ang_vel_his'][-1] = base_ang_vel
                    history_buffers['joint_pos_his'][-1] = joint_pos_diff[:23]
                    history_buffers['joint_vel_his'][-1] = qvel_seq[:23]
                    
                    offset = 0
                    obs[offset:offset + 64] = motion_latent.cpu().numpy()
                    offset += 64
                    obs[offset:offset + history_len * 3] = history_buffers['base_ang_vel_his'].reshape(-1)
                    offset += history_len * 3
                    obs[offset:offset + history_len * 23] = history_buffers['joint_pos_his'].reshape(-1)
                    offset += history_len * 23
                    obs[offset:offset + history_len * 23] = history_buffers['joint_vel_his'].reshape(-1)
                    offset += history_len * 23
                    obs[offset:offset + history_len * 23] = history_buffers['action_his'].reshape(-1)


                    obs_tensor = obs.reshape(1, -1).astype(np.float32)
                    action = policy.run(None, {'input': obs_tensor})[0]
                    action = torch.from_numpy(action)  
                    action = expand_23d_to_29d(action)


                    action_rate = np.sum((action.numpy().squeeze(0) - action_buffer) ** 2)

                    current_time = time.time() - start
                  
                    action_raw = action.numpy().squeeze(0)

                    action_buffer = action.numpy().squeeze(0)
                    history_buffers['action_his'][:-1] = history_buffers['action_his'][1:]
                    history_buffers['action_his'][-1] = action_buffer[:23]

                    target_dof_pos = action.numpy().squeeze(0) * action_scale + joint_pos_array_seq
                    target_dof_pos = target_dof_pos.reshape(-1,)
                    target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                    timestep+=1


                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            viewer.close()

            
        
        except KeyboardInterrupt:
            print("用户中断模拟")
        finally:
            print("模拟结束")


