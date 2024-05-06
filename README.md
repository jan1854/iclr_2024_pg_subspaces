# iclr2024_pg_subspaces
This repository contains the code to reproduce experiments of the ICLR 2024 paper [Identifying Policy Gradient Subspaces](https://openreview.net/pdf?id=iPWxqnt2ke) by Jan Schneider, Pierre Schumacher, Simon Guist, Le Chen,
Daniel Häufle, Bernhard Schölkopf, and Dieter Büchler.

## Installation
Download [MuJoCo version 1.50.](https://www.roboti.us/download/mjpro150_linux.zip) and extract the zip archive to ~/.mujoco.
It has to be version 1.50. since the code builds upon `stable-baselines3==1.8.0`, which depends on an old `gym` version and thus requires an old version of `mujoco-py` and consequently MuJoCo.

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
To train an agent, use the following command (the working directory should be `iclr_2024_pg_subspaces`).
```
python -m pg_subspaces.scripts.train [arg=value ...]
```
Important arguments include

| Argument              | Description                                                                                                                     | default       |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------|
| `env`                 | The environment to train on (use `dmc_{Domain}-{task}-v1` for DMC environments, e.g., `dmc_Finger-spin-v1`)                     | `Pendulum-v1` |
| `algorithm`           | The algorithm configuration to use (this includes the algorithm and hyperparameters), see `scripts/conf/algorithm` for examples | `ppo_default` |
| `checkpoint_interval` | The interval at which to store checkpoints for later analysis                                                                   | `100000`      |

To view the full list of arguments and their default values, have a look at `pg_subspaces/scripts/conf/train.yaml`.

The training script will create a log directory under `logs/training/{env}/{date}/{time}`.
Use tensorboard to view the learning curve (scalar: `eval/mean_reward`).
```
tensorboard --logdir /path/to/trainlogs
```

### Running the analysis
To run the gradient subspace fraction analysis, use the following command
```
python -m pg_subspaces.scripts.analyze train_logs=/path/to/trainlogs [arg=value ...]
```
where `/path/to/trainlogs` should be replaced by the path to the log directory created by the train command from section [Training an agent](#training-an-agent).

Important arguments include

| Argument                 | Description                                                                                                                                       | default       |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `min_interval`           | The interval at which to execute the analysis (should be a multiple of the `checkpoint_interval` used during training)                            | `100000`      |
| `num_workers`            | The number of checkpoints to analyze in parallel via multiprocessing, default                                                                     | `1`           |
| `device`                 | The device to use for analysis (`cpu` or `cuda`); analysis on a GPU is recommended                                                                | `auto`        |
| `analysis.hessian_eigen` | Which method to use for estimating the Hessian eigenvectors (`lanczos` for the Lanczos method, `explicit` for explicitly calculating the Hessian) | `lanczos`     |
To view the full list of arguments and their default values, have a look at `pg_subspaces/scripts/conf/analyze.yaml` and `pg_subspaces/scripts/conf/analysis/gradient_subspace_fraction_analysis.yaml`.

The analysis results will be added to the tensorboard logs of the training run (scalars: `high_curvature_subspace_analysis/default/gradient_subspace_fraction_*/*`).

To compute the subspace overlap metric, run the following command (need to run after the gradient subspace fraction analysis)
```
python -m pg_subspaces.scripts.compute_subspace_overlaps train_logs=/path/to/trainlogs [arg=value ...]
```

Important arguments include

| Argument                       | Description                                                                                                                                       | default                                                     |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `num_workers`                  | The number of checkpoints to analyze in parallel via multiprocessing, default                                                                     | `1`                                                         |
| `device`                       | The device to use for analysis (`cpu` or `cuda`); analysis on a GPU is recommended                                                                | `auto`                                                      |
| `analysis.hessian_eigen`       | Which method to use for estimating the Hessian eigenvectors (`lanczos` for the Lanczos method, `explicit` for explicitly calculating the Hessian) | `lanczos`                                                   |
| `eigenvec_overlap_checkpoints` | A list of initial timestep to compare the subspace to (values for *t<sub>1</sub>* in the paper)                                                   | `[0, 10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]` |
| `top_eigenvec_levels`          | A list of subspace dimensionalities (values for *k* in the paper)                                                                                 | `[1, 2, 5, 10, 20, 50, 100]`                                |
To view the full list of arguments and their default values, have a look at `pg_subspaces/scripts/conf/compute_subspace_overlaps.yaml`.

The analysis results will be added to the tensorboard logs of the training run (scalars: `high_curvature_subspace_analysis/default/overlaps_*/*`).

### Citation
```latex
@inproceedings{schneider2024identifying,
    title={Identifying Policy Gradient Subspaces},
    author={Schneider, Jan and Schumacher, Pierre and Guist, Simon and Chen, Le and H{\"a}ufle, Daniel and Sch{\"o}lkopf, Bernhard and B{\"u}chler, Dieter},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```