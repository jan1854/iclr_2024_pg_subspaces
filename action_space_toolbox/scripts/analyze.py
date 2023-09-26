import concurrent.futures
import functools
import logging
import multiprocessing
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import gym
import hydra
import omegaconf
import stable_baselines3
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import action_space_toolbox
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs
from sb3_utils.common.agent_spec import CheckpointAgentSpec

logger = logging.getLogger(__name__)


def analysis_worker(
    analysis_cfg: omegaconf.DictConfig,
    run_dir: Path,
    agent_step: int,
    device: Optional[torch.device],
    overwrite_results: bool,
    show_progress: bool,
) -> TensorboardLogs:
    train_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    if device is None:
        device = train_cfg.algorithm.algorithm.device

    env = gym.make(train_cfg.env, **train_cfg.env_args)

    agent_checkpoint = (
        run_dir / "checkpoints" / f"{train_cfg.algorithm.name}_{agent_step}_steps"
    )
    agent_spec = CheckpointAgentSpec(
        hydra.utils.get_class(train_cfg.algorithm.algorithm._target_),
        agent_checkpoint,
        device,
        agent_kwargs={"tensorboard_logs": None},
    )
    analysis = hydra.utils.instantiate(
        analysis_cfg,
        env_factory=functools.partial(gym.make, train_cfg.env, **train_cfg.env_args),
        agent_spec=agent_spec,
        run_dir=run_dir,
    )
    if hasattr(env, "base_env_timestep_factor"):
        base_env_timestep_factor = env.base_env_timestep_factor
    else:
        base_env_timestep_factor = 1
    return analysis.do_analysis(
        agent_step * base_env_timestep_factor, overwrite_results, show_progress
    )


def get_step_from_checkpoint(file_name: str) -> int:
    step_str = max(re.findall("[0-9]*", file_name))
    return int(step_str)


@hydra.main(version_base=None, config_path="conf", config_name="analyze")
def analyze(cfg: omegaconf.DictConfig) -> None:
    result_commit = subprocess.run(
        ["git", "-C", f"{Path(__file__).parent}", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    )
    logger.debug(
        f"Commit {Path(__file__).parents[2].name}: {result_commit.stdout.decode().strip()}"
    )
    train_logs = Path(cfg.train_logs)
    if not train_logs.is_absolute():
        train_logs = Path(hydra.utils.get_original_cwd()) / cfg.train_logs
    logger.info(f"Analyzing results in {train_logs}")

    experiment_dir = train_logs.parent if train_logs.name.isnumeric() else train_logs
    # assert re.match("[0-9]{2}-[0-9]{2}-[0-9]{2}", experiment_dir.name)
    train_logs_relative = train_logs.relative_to(experiment_dir.parents[3])
    if Path(cfg.log_dir).is_absolute():
        log_dir = Path(cfg.log_dir)
    else:
        log_dir = Path(hydra.utils.get_original_cwd()) / cfg.log_dir
    train_logs_local = log_dir / train_logs_relative
    if cfg.sync_train_logs:
        train_logs_local.mkdir(exist_ok=True, parents=True)
        logger.info(f"Syncing logs from {train_logs} to {train_logs_local.absolute()}.")
        rsync_result = subprocess.run(
            ["rsync", "-uaz", str(train_logs) + "/", str(train_logs_local)],
            stderr=subprocess.STDOUT,
        )
        rsync_result.check_returncode()
        logger.info("Synced logs successfully.")
    else:
        train_logs_local = train_logs

    if (train_logs_local / "checkpoints").exists():
        run_logs = [train_logs_local]
    else:
        run_logs = [
            d for d in train_logs_local.iterdir() if d.is_dir() and d.name.isdigit()
        ]

    # Determine the base_env_timestep_factor to load the correct checkpoints
    train_cfg = OmegaConf.load(run_logs[0] / ".hydra" / "config.yaml")
    env = gym.make(train_cfg.env, **train_cfg.env_args)
    if hasattr(env, "base_env_timestep_factor"):
        base_env_timestep_factor = env.base_env_timestep_factor
    else:
        base_env_timestep_factor = 1

    jobs = []
    summary_writers = {}
    for log_dir in run_logs:
        summary_writers[log_dir] = SummaryWriter(str(log_dir / "tensorboard"))
        checkpoints_dir = log_dir / "checkpoints"
        checkpoint_steps = [
            get_step_from_checkpoint(checkpoint.name)
            for checkpoint in checkpoints_dir.iterdir()
        ]
        checkpoint_steps.sort()
        checkpoints_to_analyze = []
        if cfg.checkpoints_to_analyze is not None:
            for checkpoint in cfg.checkpoints_to_analyze:
                checkpoint_no_action_repeat = (
                    int(checkpoint) // base_env_timestep_factor
                )
                if checkpoint_no_action_repeat in checkpoint_steps:
                    checkpoints_to_analyze.append(
                        (log_dir, checkpoint_no_action_repeat)
                    )
                else:
                    logger.warning(
                        f"Did not find checkpoint {checkpoint} in {checkpoints_dir}, skipping."
                    )
        else:
            for agent_step in checkpoint_steps:
                env_step = agent_step * base_env_timestep_factor

                if (
                    env_step >= cfg.first_checkpoint
                    and (cfg.last_checkpoint is None or env_step <= cfg.last_checkpoint)
                    and env_step - cfg.first_checkpoint
                    >= len(checkpoints_to_analyze) * cfg.min_interval
                ):
                    checkpoints_to_analyze.append((log_dir, agent_step))
        jobs.extend(checkpoints_to_analyze)
    jobs.sort(key=lambda j: (j[1], j[0]))

    if cfg.num_workers == 1:
        for log_dir, agent_step in tqdm(jobs, desc="Analyzing logs"):
            logs = analysis_worker(
                cfg.analysis,
                log_dir,
                agent_step,
                cfg.device,
                cfg.overwrite_results,
                show_progress=True,
            )
            logs.log(summary_writers[log_dir])
    else:
        # In contrast to multiprocessing.Pool, concurrent.futures.ProcessPoolExecutor allows nesting processes
        pool = concurrent.futures.ProcessPoolExecutor(
            cfg.num_workers, mp_context=torch.multiprocessing.get_context("spawn")
        )
        try:
            results = []
            for log_dir, agent_step in jobs:
                results.append(
                    pool.submit(
                        analysis_worker,
                        cfg.analysis,
                        log_dir,
                        agent_step,
                        cfg.device,
                        cfg.overwrite_results,
                        show_progress=False,
                    )
                )
            for result, (log_dir, agent_step) in tqdm(
                zip(results, jobs),
                total=len(jobs),
                desc="Analyzing logs",
            ):
                logs = result.result()
                logs.log(summary_writers[log_dir])
        except Exception as e:
            # TODO: Need to terminate the children's children as well
            for process in multiprocessing.active_children():
                process.terminate()
            time.sleep(1)
            raise e
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    if cfg.sync_train_logs:
        logger.info(
            f"Syncing results from {train_logs_local} back to {train_logs.absolute()}."
        )
        rsync_result = subprocess.run(
            ["rsync", "-uaz", str(train_logs_local) + "/", str(train_logs)],
            stderr=subprocess.STDOUT,
        )
        rsync_result.check_returncode()
        logger.info("Synced results successfully.")


if __name__ == "__main__":
    analyze()
