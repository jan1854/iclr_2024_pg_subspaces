import multiprocessing
import re
from pathlib import Path
from typing import Optional

import gym
import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import action_space_toolbox


def analysis_worker(
    analysis_cfg: omegaconf.DictConfig,
    log_path: Path,
    step: int,
    device: Optional[torch.device],
):
    train_cfg = OmegaConf.load(log_path / ".hydra" / "config.yaml")
    env = gym.make(train_cfg.env, **train_cfg.env_args)
    agent_class = hydra.utils.get_class(train_cfg.algorithm.algorithm._target_)
    if device is None:
        device = train_cfg.algorithm.algorithm.device
    agent = agent_class.load(
        log_path / "checkpoints" / f"{train_cfg.algorithm.name}_{step}_steps",
        env,
        device=device,
    )
    summary_writer = SummaryWriter(str(log_path / "tensorboard"))
    analysis = hydra.utils.instantiate(
        analysis_cfg, env=env, agent=agent, summary_writer=summary_writer
    )
    analysis.do_analysis(step)


def get_step_from_checkpoint(file_name: str) -> int:
    step_str = max(re.findall("[0-9]*", file_name))
    return int(step_str)


@hydra.main(version_base=None, config_path="conf", config_name="analyze")
def gradient_analysis(cfg: omegaconf.DictConfig) -> None:
    train_logs = Path(cfg.train_logs)
    if (train_logs / "checkpoints").exists():
        run_logs = [train_logs]
    else:
        run_logs = [d for d in train_logs.iterdir() if d.is_dir() and d.name.isdigit()]
    jobs = []
    for log_dir in run_logs:
        checkpoints_dir = log_dir / "checkpoints"
        checkpoint_steps = [
            get_step_from_checkpoint(checkpoint.name)
            for checkpoint in checkpoints_dir.iterdir()
        ]
        checkpoint_steps.sort()
        steps_for_analysis = []
        # TODO: Deal with multiple seeds configurations
        # TODO: Check whether it was already executed, probably one level below is better
        for step in checkpoint_steps:
            if step >= len(steps_for_analysis) * cfg.min_interval:
                steps_for_analysis.append((log_dir, step))
        jobs.extend(steps_for_analysis)
    jobs.sort(key=lambda j: (j[1], j[0]))

    pool = multiprocessing.Pool(cfg.num_workers)
    results = []
    for log_dir, step in jobs:
        results.append(
            pool.apply_async(
                analysis_worker,
                args=(cfg.analysis, log_dir, step, cfg.device),
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
