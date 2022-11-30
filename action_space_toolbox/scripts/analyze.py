import functools
import logging
import multiprocessing
import re
from pathlib import Path
from typing import Optional

import gym
import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

import action_space_toolbox


logger = logging.getLogger(__file__)


def analysis_worker(
    analysis_cfg: omegaconf.DictConfig,
    run_dir: Path,
    agent_step: int,
    device: Optional[torch.device],
    overwrite_results: bool,
):
    train_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    agent_class = hydra.utils.get_class(train_cfg.algorithm.algorithm._target_)
    if device is None:
        device = train_cfg.algorithm.algorithm.device

    env = gym.make(train_cfg.env, **train_cfg.env_args)

    agent_checkpoint = (
        run_dir / "checkpoints" / f"{train_cfg.algorithm.name}_{agent_step}_steps"
    )
    analysis = hydra.utils.instantiate(
        analysis_cfg,
        env_factory=functools.partial(gym.make, train_cfg.env, **train_cfg.env_args),
        agent_factory=functools.partial(
            agent_class.load, agent_checkpoint, env, device=device
        ),
        run_dir=run_dir,
    )
    analysis.do_analysis(agent_step * env.base_env_timestep_factor, overwrite_results)


def get_step_from_checkpoint(file_name: str) -> int:
    step_str = max(re.findall("[0-9]*", file_name))
    return int(step_str)


@hydra.main(version_base=None, config_path="conf", config_name="analyze")
def gradient_analysis(cfg: omegaconf.DictConfig) -> None:
    logger.info(f"Analyzing results in {cfg.train_logs}")
    train_logs = Path(cfg.train_logs)
    if (train_logs / "checkpoints").exists():
        run_logs = [train_logs]
    else:
        run_logs = [d for d in train_logs.iterdir() if d.is_dir() and d.name.isdigit()]

    # Determine the base_env_timestep_factor to load the correct checkpoints
    train_cfg = OmegaConf.load(run_logs[0] / ".hydra" / "config.yaml")
    env = gym.make(train_cfg.env, **train_cfg.env_args)
    base_env_timestep_factor = env.base_env_timestep_factor

    jobs = []
    for log_dir in run_logs:
        checkpoints_dir = log_dir / "checkpoints"
        checkpoint_steps = [
            get_step_from_checkpoint(checkpoint.name)
            for checkpoint in checkpoints_dir.iterdir()
        ]
        checkpoint_steps.sort()
        if cfg.checkpoints_to_analyze is not None:
            checkpoints_to_analyze = cfg.checkpoints_to_analyze
        else:
            checkpoints_to_analyze = []
            for agent_step in checkpoint_steps:
                if (
                    agent_step * base_env_timestep_factor
                    >= len(checkpoints_to_analyze) * cfg.min_interval
                ):
                    checkpoints_to_analyze.append((log_dir, agent_step))
        jobs.extend(checkpoints_to_analyze)
    jobs.sort(key=lambda j: (j[1], j[0]))

    if cfg.num_workers == 1:
        for log_dir, agent_step in jobs:
            analysis_worker(
                cfg.analysis, log_dir, agent_step, cfg.device, cfg.overwrite_results
            )
    else:
        pool = multiprocessing.get_context("spawn").Pool(cfg.num_workers)
        results = []
        for log_dir, agent_step in jobs:
            results.append(
                pool.apply_async(
                    analysis_worker,
                    args=(cfg.analysis, log_dir, agent_step, cfg.device),
                )
            )
        try:
            for result in results:
                result.get()  # Check for errors
            pool.close()
            pool.join()
        except:
            pool.terminate()
            raise


if __name__ == "__main__":
    gradient_analysis()
