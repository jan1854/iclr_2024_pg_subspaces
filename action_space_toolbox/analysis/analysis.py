from pathlib import Path

import gym
import stable_baselines3
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter


class Analysis:
    def __init__(
        self,
        analysis_name: str,
        env: gym.Env,
        agent: stable_baselines3.ppo.PPO,
        run_dir: Path,
    ):
        self.analysis_name = analysis_name
        self.env = DummyVecEnv([lambda: env])
        self.agent = agent
        self.run_dir = run_dir
        self.summary_writer = SummaryWriter(str(run_dir / "tensorboard"))
        self.analyses_log_file = run_dir / ".analyses.yaml"
        if not self.analyses_log_file.exists():
            self.analyses_log_file.touch()

    def do_analysis(self, env_step: int, overwrite: bool = False) -> None:
        print("Analysis, yo!")
        # Check whether the analysis was already done for the current step
        with self.analyses_log_file.open("r") as analyses_log_file:
            analyses_logs = yaml.safe_load(analyses_log_file)
        if analyses_logs is None:
            analyses_logs = {}
        curr_analysis_logs = analyses_logs.get(self.analysis_name, [])
        if env_step not in curr_analysis_logs or overwrite:
            self._do_analysis(env_step)
            # Safe that the analysis was done for this step
            curr_analysis_logs = sorted(curr_analysis_logs + [env_step])
            analyses_logs[self.analysis_name] = curr_analysis_logs
            with self.analyses_log_file.open("w") as analyses_log_file:
                yaml.dump(analyses_logs, analyses_log_file)

    def _do_analysis(self, env_step: int) -> None:
        pass
