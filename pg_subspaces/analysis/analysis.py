import abc
import logging

import filelock as filelock
import torch.multiprocessing
from pathlib import Path
from typing import Callable, Dict, List

import gym
import yaml

from pg_subspaces.metrics.tensorboard_logs import (
    TensorboardLogs,
    add_new_data_indicator,
)
from pg_subspaces.sb3_utils.common.agent_spec import AgentSpec

logger = logging.getLogger(__name__)

# To avoid too many open files problems when passing tensors between processes (see
# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936)
torch.multiprocessing.set_sharing_strategy("file_system")
# To avoid the warning "Forking a process while a parallel region is active is potentially unsafe."
torch.set_num_threads(1)


class Analysis(abc.ABC):
    def __init__(
        self,
        analysis_name: str,
        analysis_run_id: str,
        env_factory_or_dataset: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
    ):
        self.analysis_name = analysis_name
        self.analysis_run_id = analysis_run_id
        self.env_factory_or_dataset = env_factory_or_dataset
        self.agent_spec = agent_spec
        self.run_dir = run_dir
        self._analyses_log_file = run_dir / ".analyses.yaml"
        if not self._analyses_log_file.exists():
            self._analyses_log_file.touch()

    def do_analysis(
        self,
        env_step: int,
        overwrite_results: bool = False,
        show_progress: bool = False,
    ) -> TensorboardLogs:
        prefix = f"{self.analysis_name}/{self.analysis_run_id}"
        prefix_step_plots = f"{self.analysis_name}_step_plots/{self.analysis_run_id}"
        logs = TensorboardLogs(prefix, prefix_step_plots)
        # Check whether the analysis was already done for the current step
        analyses_logs = self._load_analysis_logs()
        curr_analysis_logs = analyses_logs[self.analysis_name][self.analysis_run_id]
        if env_step not in curr_analysis_logs or overwrite_results:
            logs = self._do_analysis(env_step, logs, overwrite_results, show_progress)
            if env_step not in curr_analysis_logs:
                # Lock the analysis.yaml file to ensure sequential access
                lock = filelock.FileLock(
                    self._analyses_log_file.with_suffix(
                        self._analyses_log_file.suffix + ".lock"
                    )
                )
                with lock.acquire(timeout=60):
                    # Re-read the analyses.yaml file in case it was changed by another process in the meantime
                    analyses_logs = self._load_analysis_logs()
                    # Save that the analysis was done for this step
                    analyses_logs[self.analysis_name][self.analysis_run_id] += [
                        env_step
                    ]
                    analyses_logs[self.analysis_name][self.analysis_run_id].sort()
                    with self._analyses_log_file.open("w") as analyses_log_file:
                        yaml.dump(analyses_logs, analyses_log_file)
            add_new_data_indicator(self.run_dir)
        return logs

    def _load_analysis_logs(self) -> Dict[str, Dict[str, List[int]]]:
        with self._analyses_log_file.open("r") as analyses_log_file:
            analyses_logs = yaml.safe_load(analyses_log_file)
        if analyses_logs is None:
            analyses_logs = {}
        if self.analysis_name not in analyses_logs:
            analyses_logs[self.analysis_name] = {}
        if self.analysis_run_id not in analyses_logs[self.analysis_name]:
            analyses_logs[self.analysis_name][self.analysis_run_id] = []
        return analyses_logs

    @abc.abstractmethod
    def _do_analysis(
        self,
        env_step: int,
        logs: TensorboardLogs,
        overwrite_results: bool,
        verbose: bool,
    ) -> TensorboardLogs:
        pass
