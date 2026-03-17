# 🤖 RoboGhost 

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://gentlefress.github.io/roboghost-proj/)
[[Arxiv]](https://arxiv.org/html/2510.14952)

## 📌 Overview

RoboGhost is a retargeting-free framework that enables language-guided humanoid control via motion latents.

Unlike traditional pipelines that require motion decoding and retargeting, RoboGhost directly conditions a diffusion-based policy on language-generated motion latents. This eliminates error-prone intermediate stages, reducing deployment latency from 17.85s to 5.84s while improving control precision.

You can command humanoids to perform complex motions using natural language, without manual tuning or retargeting.


## ⚙️ Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). 

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory),Place the whole_body_tracking under RoboGhost in the same level directory of IsaacLab:
```bash

git clone https://github.com/q2reqr/RoboGhost.git
```


- Replace **rl_cfg.py** in the Isaac Lab directory with **rl_cfg.py**  in the root directory of this repo

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## 📊  Data
- Provide a subset of the training data here: https://huggingface.co/datasets/SanQing1/RoboGhost/tree/main, which contains **`roboghost_all.pkl`** and **`general.pt`**. You can download them to any location.
## 🚀 Train

### Teacher Policy Training

- Train teacher policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \--headless --logger wandb --log_project_name {project_name} --run_name {run_name} --pkl_path /path/to/roboghost_all.pkl
```

### Student Policy Training

- Train student policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \--headless --logger wandb --log_project_name {project_name} --run_name {run_name} --dagger --resume True --resume_path /path/to/teacher_policy_ckpt --pkl_path /path/to/roboghost_all.pkl
```

## 📈 Evaluation

### Teacher Policy Evaluation

- Play the trained teacher policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=1 --resume True --resume_path /path/to/teacher_policy_ckpt --pkl_path /path/to/roboghost_all.pkl
```

### Student Policy Evaluation

- Play the trained student policy by the following command:

```bash
python scripts/rsl_rl/play_student.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=1 --dagger --resume True --resume_path /path/to/student_policy_ckpt --pkl_path /path/to/roboghost_all.pkl
```

## 📦 Export model
- Export student policy to onnx model
```bash
cd deploy/roboghost/save_onnx
python save_onnx.py --ckpt_path /path/to/student_policy_ckpt
```

## 🔄 Sim2Sim
- Install unitree-rl-gym and unitree_sdk2py  by following
  the [installation guide](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/doc/setup_en.md). 

```bash
cd deploy/roboghost
python deploy_mujoco/RoboGhost_mujoco.py --onnx_path /path/to/onnx --motion_latent_path /path/to/general.pt --motion_id {motion_id}
```

## 🦾 Sim2Real
```bash
cd deploy/roboghost
python deploy_real/RoboGhost_real.py --net {net_interface} --config /path/to/roboghost.yaml --motion_latent_path /path/to/general.pt --motion_id {motion_id}
```
- please set the network interface name to your own that connects to the robot 

## 🎉 Acknowledgments

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

RoboGhost is built upon [Beyondmimic](https://beyondmimic.github.io/).
We use publicly available human motion datasets, including [Humanml](https://github.com/EricGuo5513/HumanML3D) , and employ [GMR](https://github.com/YanjieZe/GMR) for retargeting.

[mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
[unitree\_sdk2\_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Hardware communication interface for physical deployment.

