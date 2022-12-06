import abc
import logging

import torch.multiprocessing
from pathlib import Path
from typing import Callable

import gym
import stable_baselines3
import yaml

from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


logger = logging.getLogger(__name__)


class Analysis(abc.ABC):
    def __init__(
        self,
        analysis_name: str,
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[[], stable_baselines3.ppo.PPO],
        run_dir: Path,
        num_processes: int,
    ):
        self.analysis_name = analysis_name
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.run_dir = run_dir
        self.num_processes = num_processes
        self._analyses_log_file = run_dir / ".analyses.yaml"
        if not self._analyses_log_file.exists():
            self._analyses_log_file.touch()
        self._new_data_indicator = run_dir / ".new_data"

    def do_analysis(
        self,
        env_step: int,
        overwrite_results: bool = False,
        show_progress: bool = False,
    ) -> TensorboardLogs:
        # Check whether the analysis was already done for the current step
        with self._analyses_log_file.open("r") as analyses_log_file:
            analyses_logs = yaml.safe_load(analyses_log_file)
        if analyses_logs is None:
            analyses_logs = {}
        curr_analysis_logs = analyses_logs.get(self.analysis_name, [])
        if env_step not in curr_analysis_logs or overwrite_results:
            with torch.multiprocessing.get_context("spawn").Pool(
                self.num_processes
            ) as pool:
                results = self._do_analysis(
                    pool, env_step, overwrite_results, show_progress
                )
            if env_step not in curr_analysis_logs:
                # Safe that the analysis was done for this step
                curr_analysis_logs = sorted(curr_analysis_logs + [env_step])
                analyses_logs[self.analysis_name] = curr_analysis_logs
                with self._analyses_log_file.open("w") as analyses_log_file:
                    yaml.dump(analyses_logs, analyses_log_file)
            self._new_data_indicator.touch()
            return results
        else:
            return TensorboardLogs()

    @abc.abstractmethod
    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
        env_step: int,
        overwrite_results: bool,
        verbose: bool,
    ) -> TensorboardLogs:
        pass
