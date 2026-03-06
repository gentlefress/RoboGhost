# RoboGhost [你的项目名称]

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://gentlefress.github.io/roboghost-proj/)
[[Arxiv]](https://arxiv.org/html/2510.14952)

## Overview

RoboGhost is a retargeting-free framework that enables language-guided humanoid control via motion latents.

Unlike traditional pipelines that require motion decoding and retargeting, RoboGhost directly conditions a diffusion-based policy on language-generated motion latents. This eliminates error-prone intermediate stages, reducing deployment latency from 17.85s to 5.84s while improving control precision.

You can command humanoids to perform complex motions using natural language, without manual tuning or retargeting.


## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). 

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):
```bash

git clone https://github.com/q2reqr/roboghost_open_source.git