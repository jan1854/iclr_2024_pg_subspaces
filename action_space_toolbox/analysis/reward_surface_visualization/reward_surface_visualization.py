import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Dict

import PIL
import gym
import numpy as np
import stable_baselines3.common.base_class
import stable_baselines3.common.buffers
import stable_baselines3.common.vec_env
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.reward_surface_visualization.eval_parameters import (
    eval_parameters,
)
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.sb3_training import fill_rollout_buffer, ppo_loss
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


# To avoid too many open files problems when passing tensors between processes (see
# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936)
torch.multiprocessing.set_sharing_strategy("file_system")
# To avoid the warning "Forking a process while a parallel region is active is potentially unsafe."
torch.set_num_threads(1)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    reward_undiscounted: float
    reward_discounted: float
    ppo_loss: float


class RewardSurfaceVisualization(Analysis):
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[[], stable_baselines3.ppo.PPO],
        run_dir: Path,
        grid_size: int,
        num_steps: int,
        num_plots: int,
        num_processes: int,
    ):
        super().__init__(
            "reward_surface_visualization",
            env_factory,
            agent_factory,
            run_dir,
            num_processes,
        )
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.num_plots = num_plots
        self.num_processes = num_processes
        self.out_dir = run_dir / "analyses" / "reward_surface_visualization"
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.reward_undiscounted_dir = self.out_dir / "reward_undiscounted"
        self.reward_discounted_dir = self.out_dir / "reward_discounted"
        self.loss_dir = self.out_dir / "loss"
        self.reward_undiscounted_data_dir = self.reward_undiscounted_dir / "data"
        self.reward_discounted_data_dir = self.reward_discounted_dir / "data"
        self.loss_data_dir = self.loss_dir / "data"

    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
        env_step: int,
        overwrite_results: bool,
        show_progress: bool,
    ) -> TensorboardLogs:
        logs = TensorboardLogs()
        for i in range(self.num_plots):
            if (
                not overwrite_results
                and (
                    self.loss_dir
                    / "linear"
                    / f"{self._result_filename('loss_surface', env_step, i)}.png"
                ).exists()
                and (
                    self.loss_dir
                    / "log"
                    / f"{self._result_filename('loss_surface', env_step, i)}.png"
                ).exists()
            ):
                continue
            if show_progress:
                logger.info(f"Creating plot {i} for step {env_step}.")

            agent = self.agent_factory()
            direction1 = [
                self.sample_filter_normalized_direction(p.detach())
                for p in agent.policy.parameters()
            ]
            direction2 = [
                self.sample_filter_normalized_direction(p.detach())
                for p in agent.policy.parameters()
            ]

            agent_weights = [p.data.detach() for p in agent.policy.parameters()]
            weights_offsets = [[None] * self.grid_size for _ in range(self.grid_size)]
            coords = np.linspace(-1.0, 1.0, num=self.grid_size)

            for offset1_idx, offset1_scalar in enumerate(coords):
                weights_curr_offset1 = [
                    a_weight + off * offset1_scalar
                    for a_weight, off in zip(agent_weights, direction1)
                ]
                for offset2_idx, offset2_scalar in enumerate(coords):
                    weights_curr_offsets = [
                        a_weight + off * offset2_scalar
                        for a_weight, off in zip(weights_curr_offset1, direction2)
                    ]
                    weights_offsets[offset1_idx][offset2_idx] = weights_curr_offsets

            weights_offsets_flat = [
                item for sublist in weights_offsets for item in sublist
            ]

            analysis_results_iter = process_pool.imap(
                functools.partial(
                    self.analysis_worker,
                    env_factory=self.env_factory,
                    agent_factory=self.agent_factory,
                    num_steps=self.num_steps,
                ),
                weights_offsets_flat,
            )
            analysis_results_flat = [
                r
                for r in tqdm(
                    analysis_results_iter,
                    disable=not show_progress,
                    mininterval=120,
                    total=self.grid_size**2,
                )
            ]

            rewards_undiscounted = np.array(
                [result.reward_undiscounted for result in analysis_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_undiscounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_undiscounted_file = (
                self.reward_undiscounted_data_dir
                / self._result_filename("reward_surface_undiscounted", env_step, i)
            )
            np.save(str(rewards_undiscounted_file), rewards_undiscounted)

            rewards_discounted = np.array(
                [result.reward_discounted for result in analysis_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_discounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_discounted_file = (
                self.reward_discounted_data_dir
                / self._result_filename("rewards_discounted", env_step, i)
            )
            np.save(str(rewards_discounted_file), rewards_discounted)

            loss = np.array(
                [result.ppo_loss for result in analysis_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.loss_data_dir.mkdir(parents=True, exist_ok=True)
            loss_file = self.loss_data_dir / self._result_filename("loss", env_step, i)
            np.save(str(loss_file), loss)

            # Plotting needs to happen in a separate process since matplotlib is not thread safe (see
            # https://matplotlib.org/3.1.0/faq/howto_faq.html#working-with-threads)
            logs.update(
                process_pool.apply(
                    functools.partial(
                        self.plot_worker,
                        coords,
                        rewards_undiscounted,
                        rewards_discounted,
                        loss,
                        env_step,
                        i,
                    )
                )
            )
        return logs

    @staticmethod
    def analysis_worker(
        agent_weights: Sequence[torch.Tensor],
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[[], stable_baselines3.ppo.PPO],
        num_steps: int,
    ) -> AnalysisResult:
        env = stable_baselines3.common.vec_env.DummyVecEnv([env_factory])
        agent = agent_factory()
        rollout_buffer = stable_baselines3.common.buffers.RolloutBuffer(
            num_steps,
            agent.observation_space,
            agent.action_space,
            "cpu",
            agent.gae_lambda,
            agent.gamma,
        )
        rollout_buffer_no_value_bootstrap = (
            stable_baselines3.common.buffers.RolloutBuffer(
                num_steps,
                agent.observation_space,
                agent.action_space,
                "cpu",
                agent.gae_lambda,
                agent.gamma,
            )
        )
        with torch.no_grad():
            for parameters, weights in zip(agent.policy.parameters(), agent_weights):
                parameters.data[:] = weights
        fill_rollout_buffer(
            agent,
            env,
            rollout_buffer,
            rollout_buffer_no_value_bootstrap,
            show_progress=False,
        )
        episode_rewards_undiscounted = []
        episode_rewards_discounted = []
        curr_reward_undiscounted = None
        curr_reward_discounted = None
        curr_episode_length = 0
        for episode_start, reward in zip(
            rollout_buffer_no_value_bootstrap.episode_starts,
            rollout_buffer_no_value_bootstrap.rewards,
        ):
            if episode_start:
                if curr_episode_length > 0:
                    episode_rewards_undiscounted.append(curr_reward_undiscounted)
                    episode_rewards_discounted.append(curr_reward_discounted)
                curr_episode_length = 0
                curr_reward_discounted = 0.0
                curr_reward_undiscounted = 0.0
            curr_episode_length += 1
            curr_reward_undiscounted += reward
            curr_reward_discounted += agent.gamma ** (curr_episode_length - 1) * reward
        # Since there is no next transition in the buffer, we cannot know if the last episode is complete, so only add
        # the episode if it has the maximum episode length.
        if curr_episode_length == get_episode_length(env.envs[0]):
            episode_rewards_undiscounted.append(curr_reward_undiscounted)
            episode_rewards_discounted.append(curr_reward_discounted)
        mean_episode_reward_undiscounted = np.mean(episode_rewards_undiscounted)
        mean_episode_reward_discounted = np.mean(episode_rewards_discounted)
        loss = ppo_loss(agent, next(rollout_buffer.get()))
        return AnalysisResult(
            mean_episode_reward_undiscounted,
            mean_episode_reward_discounted,
            loss.item(),
        )

    def plot_worker(
        self,
        coords: np.ndarray,
        rewards_undiscounted: np.ndarray,
        rewards_discounted: np.ndarray,
        loss: np.ndarray,
        env_step: int,
        plot_nr: int,
    ) -> TensorboardLogs:
        logs = TensorboardLogs()
        self.plot_results(
            coords,
            rewards_undiscounted,
            "reward_surface_undiscounted",
            env_step,
            plot_nr,
            "reward surface (undiscounted)",
            logs,
            self.reward_undiscounted_dir,
        )
        self.plot_results(
            coords,
            rewards_discounted,
            "reward_surface_discounted",
            env_step,
            plot_nr,
            "reward surface (discounted)",
            logs,
            self.reward_discounted_dir,
        )
        self.plot_results(
            coords,
            loss,
            "loss_surface",
            env_step,
            plot_nr,
            "loss surface",
            logs,
            self.loss_dir,
        )
        return logs

    def plot_results(
        self,
        coords: np.ndarray,
        results: np.ndarray,
        plot_name: str,
        env_step: int,
        plot_nr: int,
        title_descr: str,
        logs: TensorboardLogs,
        out_dir: Path,
    ) -> None:
        x_coords, y_coords = np.meshgrid(coords, coords)
        plot_outpath_linear = (
            out_dir
            / "linear"
            / f"{self._result_filename(plot_name, env_step, plot_nr)}.png"
        )
        plot_outpath_linear.parent.mkdir(parents=True, exist_ok=True)
        title_linear = f"{self.env_factory().spec.id} {title_descr}"
        self._plot_surface(
            x_coords,
            y_coords,
            results,
            plot_outpath_linear,
            title_linear,
            logscale=False,
        )
        plot_outpath_log = (
            out_dir
            / "log"
            / f"{self._result_filename(plot_name, env_step, plot_nr)}.png"
        )
        plot_outpath_log.parent.mkdir(parents=True, exist_ok=True)
        title_log = f"{self.env_factory().spec.id} {title_descr}"
        self._plot_surface(
            x_coords,
            y_coords,
            results,
            plot_outpath_log,
            title_log,
            logscale=True,
        )
        with Image.open(plot_outpath_linear) as im:
            # Make the image smaller so that it fits better in tensorboard
            im = im.resize(
                (im.width // 2, im.height // 2), PIL.Image.Resampling.LANCZOS
            )
            logs.add_image(f"{plot_name}/linear/{plot_nr}", im, env_step)
        with Image.open(plot_outpath_log) as im:
            # Make the image smaller so that it fits better in tensorboard
            im = im.resize(
                (im.width // 2, im.height // 2), PIL.Image.Resampling.LANCZOS
            )
            logs.add_image(f"{plot_name}/log/{plot_nr}", im, env_step)

    @staticmethod
    def sample_filter_normalized_direction(param: torch.Tensor) -> torch.Tensor:
        ndims = len(param.shape)
        if ndims == 1 or ndims == 0:
            # don't do any random direction for scalars
            return torch.zeros_like(param)
        elif ndims == 2:
            direction = torch.normal(0.0, 1.0, size=param.shape, device=param.device)
            direction /= torch.sqrt(
                torch.sum(torch.square(direction), dim=0, keepdim=True)
            )
            direction *= torch.sqrt(torch.sum(torch.square(param), dim=0, keepdim=True))
            return direction
        elif ndims == 4:
            direction = torch.normal(0.0, 1.0, size=param.shape, device=param.device)
            direction /= torch.sqrt(
                torch.sum(torch.square(direction), dim=(0, 1, 2), keepdim=True)
            )
            direction *= torch.sqrt(
                torch.sum(torch.square(param), dim=(0, 1, 2), keepdim=True)
            )
            return direction
        else:
            raise ValueError(
                f"Only 1, 2, 4 dimensional filters allowed, got {param.shape}."
            )

    @staticmethod
    def _plot_surface(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        values: np.ndarray,
        outpath: Path,
        title: Optional[str] = None,
        logscale: bool = False,
        magnitude: float = 1.0,
    ) -> None:
        fig = plt.figure()
        ax = Axes3D(fig)

        if title is not None:
            fig.suptitle(title)

        if np.min(values) < -1e9 and not logscale:
            print(
                "Warning: Data includes extremely large negative rewards ({:3E}).\
                          Consider setting logscale=True".format(
                    np.min(values)
                )
            )

        # Scale X and Y values by the step size magnitude
        x_coords = magnitude * x_coords
        y_coords = magnitude * y_coords

        real_values = values.copy()
        # Take numerically stable log of data
        if logscale:
            values_neg = values[values < 0]
            values_pos = values[values >= 0]
            values_neg = -np.log10(1 - values_neg)
            values_pos = np.log10(1 + values_pos)
            values[values < 0] = values_neg
            values[values >= 0] = values_pos

        # Plot surface
        surf = ax.plot_surface(
            x_coords,
            y_coords,
            values,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            zorder=5,
        )

        # Plot center line above surface
        Z_range = abs(np.max(values) - np.min(values))
        zline_above = np.linspace(
            values[len(values) // 2][len(values[0]) // 2],
            np.max(values) + (Z_range * 0.1),
            4,
        )
        xline_above = 0 * zline_above
        yline_above = 0 * zline_above
        ax.plot3D(xline_above, yline_above, zline_above, "black", zorder=10)

        # Plot center line below surface
        zline_below = np.linspace(
            values[len(values) // 2][len(values[0]) // 2],
            np.min(values) - (Z_range * 0.1),
            4,
        )
        xline_below = 0 * zline_below
        yline_below = 0 * zline_below
        ax.plot3D(xline_below, yline_below, zline_below, "black", zorder=0)

        # Fix colorbar labeling for log scale
        if logscale:
            # Find the highest order of magnitude
            max_Z = np.max(real_values)
            if max_Z < 0:
                max_magnitude = -math.floor(math.log10(-max_Z))
            else:
                max_magnitude = math.floor(math.log10(max_Z))

            # Find the lowest order of magnitude
            min_Z = np.min(real_values)
            if min_Z < 0:
                min_magnitude = -math.floor(math.log10(-min_Z))
            else:
                if min_Z == 0:
                    min_Z += 0.0000001
                min_magnitude = math.floor(math.log10(min_Z))

            # Create colorbar
            continuous_labels = np.round(
                np.linspace(min_magnitude, max_magnitude, 8, endpoint=True)
            )
            cbar = fig.colorbar(
                surf, shrink=0.5, aspect=5, ticks=continuous_labels, pad=0.1
            )
            cbar.set_ticks(list())

            # Manually set colorbar and z axis tick text
            zticks = []
            ztick_labels = []
            bounds = cbar.ax.get_ybound()
            for index, label in enumerate(continuous_labels):
                x = 6.0
                y = bounds[0] + (bounds[1] - bounds[0]) * index / 8

                # Format label
                zticks.append(label)
                if label > 2 or label < -2:
                    label = (
                        "$-10^{" + str(int(-label)) + "}$"
                        if label < 0
                        else "$10^{" + str(int(label)) + "}$"
                    )
                else:
                    label = (
                        "${}$".format(-(10.0 ** (-label)))
                        if label < 0
                        else "${}$".format(10.0**label)
                    )
                cbar.ax.text(x, y, label)
                ztick_labels.append("    " + label)
            ax.set_zticks(zticks)
            ax.set_zticklabels(ztick_labels)
        else:
            fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.05)

        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _result_filename(plot_name: str, env_step: int, plot_idx: int) -> str:
        return f"{plot_name}_{env_step:07d}_{plot_idx:02d}"
