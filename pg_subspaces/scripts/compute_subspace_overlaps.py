import concurrent.futures
import functools
import logging
import multiprocessing
import re
import subprocess
import time
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from pg_subspaces.analysis.high_curvature_subspace_analysis.subspace_overlaps import (
    SubspaceOverlaps,
)
from pg_subspaces.metrics.tensorboard_logs import TensorboardLogs
from pg_subspaces.sb3_utils.common.agent_spec import HydraAgentSpec
from pg_subspaces.sb3_utils.common.env.make_env import make_vec_env
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen_lanczos import (
    HessianEigenLanczos,
)
from pg_subspaces.scripts.analyze import find_parent_with_name_pattern
from pg_subspaces.utils.hydra import register_custom_resolvers

logger = logging.getLogger(__name__)


register_custom_resolvers()


def analysis_worker(
    analysis_cfg: omegaconf.DictConfig,
    run_dir: Path,
) -> TensorboardLogs:
    print("Created analysis_worker.")
    train_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    agent_spec = HydraAgentSpec(
        train_cfg.algorithm,
        "cpu",
        functools.partial(make_vec_env, train_cfg),
        None,
        agent_kwargs={"tensorboard_log": None},
    )
    subspace_overlaps = SubspaceOverlaps(
        agent_spec,
        run_dir,
        analysis_cfg.analysis_run_id,
        analysis_cfg.top_eigenvec_levels,
        HessianEigenLanczos(1e-3, 100, None),
        analysis_cfg.eigenvec_overlap_checkpoints,
        analysis_cfg.verbose,
    )
    print("Starting analysis.")
    return subspace_overlaps.analyze_subspace_overlaps()


def get_step_from_checkpoint(file_name: str) -> int:
    step_str = max(re.findall("[0-9]*", file_name))
    return int(step_str)


@hydra.main(
    version_base=None, config_path="conf", config_name="compute_subspace_overlaps"
)
def compute_subspace_overlaps(cfg: omegaconf.DictConfig) -> None:
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

    if cfg.num_workers == 1:
        for run_dir in run_dirs:
            analysis_worker(
                cfg,
                run_dir,
            )
    else:
        pool = concurrent.futures.ProcessPoolExecutor(
            cfg.num_workers, mp_context=torch.multiprocessing.get_context("spawn")
        )
        try:
            for _ in tqdm(
                pool.map(functools.partial(analysis_worker, cfg), run_dirs),
                total=len(run_dirs),
            ):
                pass
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
    compute_subspace_overlaps()
