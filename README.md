<div align="center">

<!-- 项目图标 -->
<img src="fig/icon.png" width="120px" alt="RoboGhost Icon"/>

# From Language To Locomotion Retargeting-free Humanoid Control via Motion Latent Guidance

**Accepted by ICLR 2026**

[![Website](https://img.shields.io/badge/Project-Website-8A2BE2)](https://gentlefress.github.io/roboghost-proj/)
[![Arxiv](https://img.shields.io/badge/arXiv-2510.14952-B31B1B.svg)](https://arxiv.org/html/2510.14952)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](./LICENSE)

<img src="fig/roboghost.png" width="100%"/>

</div>

## 🗞️ News
- **`2026-03-17`**: 🔥🔥🔥 We released the code of **RoboGhost**! Welcome to Star!⭐
- **`2026-02-21`**: 🔥🔥 Related work **RoboPerform** gets accepted to **CVPR 2026**! See you in Danver!
- **`2026-01-23`**: 🔥🔥 **RoboGhost** gets accepted to **ICLR 2026**! See you in Rio, Brazil!
- **`2025-12-04`**: 🔥 We released our [Project Page](https://gentlefress.github.io/RoboMirror-proj/) of **RoboMirror**.
- **`2025-12-04`**: 🔥 We released our [Project Page](https://gentlefress.github.io/RoboPerform-proj/) of **RoboPerform**.
- **`2025-10-02`**: 🔥 We released our [Project Page](https://gentlefress.github.io/roboghost-proj/) of **RoboGhost**.


## 🎯 TODO
- [x] Release the training codes and inference codes of RoboGhost.
- [x] Release the project page of RoboGhost.



## 📌 Overview

**RoboGhost** is a retargeting-free framework that enables language-guided humanoid control via motion latents. 

Unlike traditional pipelines that require complex motion decoding and manual retargeting, RoboGhost directly conditions a diffusion-based policy on language-generated **motion latents**. This eliminates error-prone intermediate stages, reducing deployment latency from **17.85s to 5.84s** while significantly improving control precision.

You can command humanoids to perform complex motions using natural language, without the need for manual joint tuning or retargeting for different robot morphologies.

## 🏗️ Framework

<div align="center">
<img src="fig/framework.png" width="100%"/>
</div>

---

## ⚙️ Installation

1.  **Install Isaac Lab v2.1.0**: Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2.  **Clone the Repository**: Place `whole_body_tracking` under `RoboGhost` in the same level directory as `IsaacLab`:
    ```bash
    git clone https://github.com/gentlefress/RoboGhost.git
    ```

3.  **Configuration Patch**:
    Replace `rl_cfg.py` in your Isaac Lab directory with the `rl_cfg.py` located in the root of this repository to ensure compatibility.

4.  **Install Library**:
    Using the Python interpreter associated with your Isaac Lab installation:
    ```bash
    python -m pip install -e source/whole_body_tracking
    ```

## 📊 Data
Download the training data from [Hugging Face](https://huggingface.co/datasets/SanQing1/RoboGhost/tree/main):
- **`roboghost_all.pkl`**: Motion dataset.
- **`general.pt`**: Pre-trained motion latents.

---

## 🚀 Training

### 1. Teacher Policy Training
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --headless --logger wandb \
  --log_project_name {project_name} \
  --run_name {run_name} \
  --pkl_path /path/to/roboghost_all.pkl
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

## 🔖 License

This project is licensed under the [BSD 3-Clause License](./LICENSE):
1. The original copyright notice must be retained.
2. The project name or organization name may not be used for promotion.
3. Any modifications must be disclosed.

For details, please read the full [LICENSE file](./LICENSE).

If you find our code or paper helpful, please consider starring our repository and citing:
```
@article{li2025language,
  title={From language to locomotion: Retargeting-free humanoid control via motion latent guidance},
  author={Li, Zhe and Chi, Cheng and Wei, Yangyang and Zhu, Boan and Peng, Yibo and Huang, Tao and Wang, Pengwei and Wang, Zhongyuan and Zhang, Shanghang and Xu, Chang},
  journal={arXiv preprint arXiv:2510.14952},
  year={2025}
}
```
