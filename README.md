# iclr2024_pg_subspaces
This repository contains the code to reproduce experiments of the ICLR submission "Identifying Policy Gradient Subspaces".

## Installation
Download [MuJoCo version 1.50.](https://www.roboti.us/download/mjpro150_linux.zip) and extract the zip archive to ~/.mujoco.
It has to be version 1.50. since stable-baselines3 depends on an old gym version, which in turn requires an old version of `mujoco-py` and consequently MuJoCo.

Make sure that the system packages `libosmesa6-dev` and `patchelf` are installed (required to build `mujoco-py`).
```
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```
Install the `pg_subspaces` package.
```
cd iclr_2024_pg_subspaces
pip install .
```

## Usage

### Training an agent
To train an agent, use the following command
```
python -m pg_subspaces.scripts.train [arg=value ...]
```
Important arguments include

| Argument              | Description                                                                                                                     | default       |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------|
| `env`                 | The environment to train on (use `dmc_{Domain}-{task}-v1` for DMC environments, e.g., `dmc_Finger-spin-v1`)                     | `Pendulum-v1` |
| `algorithm`           | The algorithm configuration to use (this includes the algorithm and hyperparameters), see `scripts/conf/algorithm` for examples | `ppo_default` |
| `checkpoint_interval` | The interval at which to store checkpoints for later analysis, default:                                                         | `100000`      |

The training script will create a log directory under `iclr2024_pg_subspaces/logs/training/{env}/{date}/{time}`.
Use tensorboard to view the learning curve.
```
tensorboard --logdir /path/to/trainlogs
```

### Running the analysis
To run the analysis, use the following command
```
python -m pg_subspaces.scripts.analyze train_logs=/path/to/trainlogs [arg=value ...]
```
where `/path/to/trainlogs` should be replaced by the path to the log directory created by the train command from section [Training an agent](#Training an agent).

Important arguments include

| Argument                 | Description                                                                                                                                       | default       |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `min_interval`           | The interval at which to execute the analysis (should be a multiple of the `checkpoint_interval` used during training)                            | `100000`      |
| `num_workers`            | The number of checkpoints to analyze in parallel via multiprocessing, default                                                                     | `1`           |
| `device`                 | The device to use for analysis (`cpu` or `cuda`); analysis on a GPU is recommended                                                                | `auto`        |
| `analysis.hessian_eigen` | Which method to use for estimating the Hessian eigenvectors (`lanczos` for the Lanczos method, `explicit` for explicitly calculating the Hessian) | `lanczos`     |

The analysis results will be added to the tensorboard logs of the training run.
