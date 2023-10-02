# action-space-optimization
This repository implements wrappers to change the control mode of reinforcement learning benchmark tasks.
The implementation supports tasks from OpenAI Gym and the DeepMind Control Suite.
Currently, the supported control modes are torque control (the regular control mode of the benchmarks), velocity control, and position control.

## Installation
Download [MuJoCo version 1.50.](https://www.roboti.us/download/mjpro150_linux.zip) and extract the zip archive to ~/.mujoco.
It has to be version 1.50. since stable-baselines3 depends on an old gym version, which in turn requires an old version of `mujoco-py` and consequently MuJoCo.

Make sure that the system packages `libosmesa6-dev` and `patchelf` are installed (required to build `mujoco-py`).
```
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```
Install poetry 
```
curl -sSL https://install.python-poetry.org | python3 -
```
Install the `pg_subspaces` package.
```
cd iclr_2024_pg_subspaces
pip install .
```

## Usage
