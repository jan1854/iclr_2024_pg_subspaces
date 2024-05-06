import concurrent.futures
import functools
import logging
import multiprocessing
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence, List, Tuple

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pg_subspaces.metrics.tensorboard_logs import TensorboardLogs
from pg_subspaces.offline_rl.load_env_dataset import load_env_dataset
from pg_subspaces.offline_rl.offline_algorithm import OfflineAlgorithm
from pg_subspaces.sb3_utils.common.agent_spec import (
    CheckpointAgentSpec,
    get_checkpoint_path,
)
from pg_subspaces.sb3_utils.common.env.make_env import make_vec_env
from pg_subspaces.utils.hydra import register_custom_resolvers

logger = logging.getLogger(__name__)


def create_jobs(
    run_dirs: Sequence[Path],
    checkpoints_to_analyze: Optional[Sequence[int]],
    min_interval: Optional[int],
    first_checkpoint: int,
    last_checkpoint: Optional[int],
) -> List[Tuple[Path, int]]:
    jobs = []
    for log_dir in run_dirs:
        checkpoints_dir = log_dir / "checkpoints"
        checkpoint_steps = [
            get_step_from_checkpoint(checkpoint.name)
            for checkpoint in checkpoints_dir.iterdir()
        ]
        checkpoint_steps.sort()
        curr_jobs = []
        if checkpoints_to_analyze is not None:
            for checkpoint in checkpoints_to_analyze:
                checkpoint_no_action_repeat = int(checkpoint)
                if checkpoint_no_action_repeat in checkpoint_steps:
                    curr_jobs.append((log_dir, checkpoint_no_action_repeat))
                else:
                    logger.warning(
                        f"Did not find checkpoint {checkpoint} in {checkpoints_dir}, skipping."
                    )
        else:
            for step in checkpoint_steps:
                if (
                    step >= first_checkpoint
                    and (last_checkpoint is None or step <= last_checkpoint)
                    and step - first_checkpoint >= len(curr_jobs) * min_interval
                ):
                    curr_jobs.append((log_dir, step))
        jobs.extend(curr_jobs)
    jobs.sort(key=lambda j: (j[1], j[0]))
    return jobs


def analysis_worker(
    analysis_cfg: omegaconf.DictConfig,
    run_dir: Path,
    agent_step: int,
    device: Optional[torch.device],
    overwrite_results: bool,
    show_progress: bool,
) -> TensorboardLogs:
    logger.debug("Created analysis_worker.")
    train_cfg_path = run_dir / ".hydra" / "config.yaml"
    # If we're analyzing the output of a tuning run, the .hydra directory is one directory up
    if not train_cfg_path.exists() and run_dir.parent.name.isnumeric():
        train_cfg_path = run_dir.parent / ".hydra" / "config.yaml"
    train_cfg = OmegaConf.load(train_cfg_path)
    if device is None:
        device = train_cfg.algorithm.algorithm.device

    agent_class = hydra.utils.get_class(train_cfg.algorithm.algorithm._target_)
    agent_spec = CheckpointAgentSpec(
        agent_class,
        run_dir / "checkpoints",
        agent_step,
        device,
        agent_kwargs={"tensorboard_logs": None},
        freeze_vec_normalize=True,
    )
    if issubclass(agent_class, OfflineAlgorithm):
        _, env_factory_or_dataset = load_env_dataset(train_cfg.logs_dataset, device)
    else:
        vec_normalize_path = get_checkpoint_path(
            run_dir / "checkpoints", agent_step, "vecnormalize"
        )
        if not vec_normalize_path.exists():
            vec_normalize_path = None
        env_factory_or_dataset = functools.partial(
            make_vec_env, train_cfg, vec_normalize_path, True
        )
    analysis = hydra.utils.instantiate(
        analysis_cfg,
        env_factory_or_dataset=env_factory_or_dataset,
        agent_spec=agent_spec,
        run_dir=run_dir,
    )
    logger.debug("Starting analysis.")
    return analysis.do_analysis(agent_step, overwrite_results, show_progress)


def get_step_from_checkpoint(file_name: str) -> int:
    return max([int(m[:-6]) for m in re.findall("[0-9]*_steps", file_name)])


def find_parent_with_name_pattern(
    child: Path, parent_name_pattern: str
) -> Optional[Path]:
    parent_name_pattern = re.compile(parent_name_pattern)
    for parent in child.parents:
        if re.fullmatch(parent_name_pattern, parent.name):
            return parent
    return None


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

    env_path = find_parent_with_name_pattern(train_logs, ".+-v[0-9]+")
    train_logs_relative = train_logs.relative_to(env_path.parents[1])
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
        run_dirs = [train_logs_local]
    else:
        run_dirs = [
            d for d in train_logs_local.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        for run_dir in run_dirs.copy():
            sub_run_dirs = [
                d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if len(sub_run_dirs) > 0:
                run_dirs.remove(run_dir)
            run_dirs.extend(sub_run_dirs)

    jobs = create_jobs(
        run_dirs,
        cfg.get("checkpoints_to_analyze"),
        cfg.get("min_interval"),
        cfg.get("first_checkpoint"),
        cfg.get("last_checkpoint"),
    )
    summary_writers = {}
    for log_dir in run_dirs:
        summary_writers[log_dir] = SummaryWriter(str(log_dir / "tensorboard"))

    if cfg.num_workers == 1:
        for log_dir, step in tqdm(jobs, desc="Analyzing logs"):
            logs = analysis_worker(
                cfg.analysis,
                log_dir,
                step,
                cfg.device,
                cfg.overwrite_results,
                show_progress=True,
            )
            if logs is not None:
                logs.log(summary_writers[log_dir])
    else:
        # In contrast to multiprocessing.Pool, concurrent.futures.ProcessPoolExecutor allows nesting processes
        pool = concurrent.futures.ProcessPoolExecutor(
            cfg.num_workers, mp_context=torch.multiprocessing.get_context("spawn")
        )
        try:
            results = []
            for log_dir, step in jobs:
                results.append(
                    pool.submit(
                        analysis_worker,
                        cfg.analysis,
                        log_dir,
                        step,
                        cfg.device,
                        cfg.overwrite_results,
                        show_progress=False,
                    )
                )
            for result, (log_dir, step) in tqdm(
                zip(results, jobs),
                total=len(jobs),
                desc="Analyzing logs",
            ):
                logs = result.result()
                if logs is not None:
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
    register_custom_resolvers()
    analyze()
