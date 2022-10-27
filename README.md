# action-space-optimization
This repository implements wrappers to change the control mode of reinforcement learning benchmark tasks.
The implementation supports tasks from OpenAI Gym and the DeepMind Control Suite.
Currently, the supported control modes are torque control (the regular control mode of the benchmarks), velocity control, and position control.

## Usage
All supported environments are registered in gym and can be created with `gym.make()`.
The environment identifier is constructed as `{original_env_id}_{control_mode_id}-v{version}`.
The control_mode_id is TC for torque control, PC for position control, and VC for velocity control.
For dm-control environments, the original_env_id is `dmc_{domain}-{task}`and the version is 1.

Instantiating a position-controlled version of gym's Pendulum:
```
import action_space_toolbox
import gym

env = gym.make("Pendulum_PC-v1")
```

Instantiating a velocity-controlled version of dm_controls Pendulum-swingup:
```
import action_space_toolbox
import gym

env = gym.make("dmc_Pendulum-swingup_VC-v1")
```

## Supported environments
OpenAI Gym: Pendulum-v1, Ant-v3, HalfCheetah-v3, Hopper-v3, Reacher-v2, Walker2d-v3

DeepMind Control Suite: All benchmarking environments except for fish and ball-in-cup environments