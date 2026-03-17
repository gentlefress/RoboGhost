"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--dagger", action="store_true", default=False, help="Switch RL Dagger")
parser.add_argument("--resume_path", type=str, default=None, help="teacher path")
parser.add_argument("--pkl_path", type=str, required=True, help="The name of the wand registry.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

keyboard_listener = None

def _init_global_keyboard_listener(global_motion_command):
    global keyboard_listener  
    
    try:
        from pynput import keyboard
        
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    if key.char == '+' or key.char == '=':
                        _next_motion(global_motion_command)
                    elif key.char == '-':
                        _previous_motion(global_motion_command)
                elif key == keyboard.Key.esc:
                    _exit_program()
            except Exception as e:
                print(f"error: {e}")
        
        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()
        print("'+' or '='next motion, '-'previous motion, 'ESC'kill")
        print("keyboard_listener start")
        
    except ImportError:
        print("need to: pip install pynput")

def _next_motion(global_motion_command):
    # global global_motion_command
    if global_motion_command is not None:
        current_ids = global_motion_command.motion_ids.clone()
        new_ids = (current_ids + 1) % global_motion_command.motion.num_motions
        global_motion_command.motion_ids[:] = new_ids
        global_motion_command.time_step = 0
        print(f"[KEYBOARD] Motion IDs increased: {current_ids.cpu().numpy()} -> {new_ids.cpu().numpy()}")
    else:
        print("MotionCommand init failed")

def _previous_motion(global_motion_command):
    # global global_motion_command
    if global_motion_command is not None:
        current_ids = global_motion_command.motion_ids.clone()
        new_ids = (current_ids - 1) % global_motion_command.motion.num_motions
        global_motion_command.motion_ids[:] = new_ids
        global_motion_command.time_step = 0
        print(f"[KEYBOARD] Motion IDs decreased: {current_ids.cpu().numpy()} -> {new_ids.cpu().numpy()}")
    else:
        print("MotionCommand init failed")

def _exit_program():
    global keyboard_listener
    print("exiting program...")
    if keyboard_listener:
        keyboard_listener.stop()
    import sys
    sys.exit()


def get_wo_hands_observation(history_obs):
    HISTORY_LEN = 15
    BASE_ANG_VEL_DIM = 3

    JOINT_POS_DIM = 29
    JOINT_VEL_DIM = 29
    ACTIONS_DIM = 29
    JOINT_POS_WO_HANDS_DIM = 23
    JOINT_VEL_WO_HANDS_DIM = 23
    ACTIONS_WO_HANDS_DIM = 23
    base_ang_vel_total_dim = BASE_ANG_VEL_DIM * HISTORY_LEN  
    joint_pos_total_dim = JOINT_POS_DIM * HISTORY_LEN  
    joint_vel_total_dim = JOINT_VEL_DIM * HISTORY_LEN  
    actions_total_dim = ACTIONS_DIM * HISTORY_LEN  
    joint_pos_total_dim_wo_hands = JOINT_POS_WO_HANDS_DIM * HISTORY_LEN
    joint_vel_total_dim_wo_hands = JOINT_VEL_WO_HANDS_DIM * HISTORY_LEN
    actions_total_dim_wo_hands = ACTIONS_WO_HANDS_DIM * HISTORY_LEN
    offsets = [0]
    dims = [
        base_ang_vel_total_dim, 
        joint_pos_total_dim,   
        joint_vel_total_dim,   
        actions_total_dim,    
    ]
    
    for d in dims:
        offsets.append(offsets[-1] + d)
    
    
    base_ang_vel = history_obs[:, offsets[0]:offsets[1]]
    joint_pos = history_obs[:, offsets[1]:offsets[2]]
    joint_vel = history_obs[:, offsets[2]:offsets[3]]
    actions = history_obs[:, offsets[3]:offsets[4]]
    
    base_ang_vel_reshaped = base_ang_vel.view(-1, HISTORY_LEN, BASE_ANG_VEL_DIM)
    joint_pos_reshaped = joint_pos.view(-1, HISTORY_LEN, JOINT_POS_DIM)
    joint_vel_reshaped = joint_vel.view(-1, HISTORY_LEN, JOINT_VEL_DIM)
    actions_reshaped = actions.view(-1, HISTORY_LEN, ACTIONS_DIM)
    joint_pos_wo_hands = joint_pos_reshaped[:, :, :-6]
    joint_vel_wo_hands = joint_vel_reshaped[:, :, :-6]
    actions_wo_hands = actions_reshaped[:, :, :-6]

    history_obs_wo_hands = torch.cat([
        base_ang_vel,  # [B, 45] 
        joint_pos_wo_hands.reshape(-1, joint_pos_total_dim_wo_hands),  # [B, 345]
        joint_vel_wo_hands.reshape(-1, joint_vel_total_dim_wo_hands),  # [B, 345]
        actions_wo_hands.reshape(-1, actions_total_dim_wo_hands),      # [B, 345]
    ], dim=-1)  # [B, 1080]

    return history_obs_wo_hands



def append_23_to_29_actions(num_envs, actions_23d):
    zeros_3d = torch.zeros(num_envs, 6, device=actions_23d.device)  # 19-21
    actions_29d = torch.cat([
        actions_23d,        
        zeros_3d,     
    ], dim=1)
    return actions_29d

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.pkl_path is not None:
            # print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.pkl_path

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = args_cli.pkl_path

    else:
        resume_path = args_cli.resume_path
        if args_cli.pkl_path is not None:
            # print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.pkl_path
        
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    global_motion_command = env.command_manager.get_term('motion')
    _init_global_keyboard_listener(global_motion_command)
    # MotionCommand.motion_ids 

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    # import ipdb;ipdb.set_trace()
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load_student(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_student_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    obs, extras = env.get_observations()
    obs_motion = extras['observations']['motion_latent']
    history_obs = get_wo_hands_observation(extras['observations']['student'])
    obs = torch.cat([obs_motion, history_obs], dim=-1)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            actions = append_23_to_29_actions(args_cli.num_envs, actions)
            obs, _, _, extras = env.step(actions)
            obs_motion = extras['observations']['motion_latent']
            history_obs = get_wo_hands_observation(extras['observations']['student'])
            obs = torch.cat([obs_motion, history_obs], dim=-1)


        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
