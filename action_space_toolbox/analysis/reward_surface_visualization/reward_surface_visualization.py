import functools
import math
import multiprocessing
from pathlib import Path
from typing import Callable, Optional

import gym
import numpy as np
import stable_baselines3.common.base_class
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.reward_surface_visualization.eval_parameters import (
    eval_parameters,
)


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
            "reward_surface_visualization", env_factory, agent_factory, run_dir
        )
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.num_plots = num_plots
        self.num_processes = num_processes
        self.out_dir = run_dir / "analyses" / "reward_surface_visualization"
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir = self.out_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    def _do_analysis(self, env_step: int) -> None:
        for i in range(self.num_plots):
            if (self.out_dir / f"{self._result_filename(env_step, i)}.png").exists():
                continue
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
            weights_offsets = [[None] * (self.grid_size + 1)] * (self.grid_size + 1)

            for offset_idx1 in range(self.grid_size + 1):
                offset1_scalar = (self.grid_size // 2 - offset_idx1) / (
                    self.grid_size // 2
                )
                weights_curr_offset1 = [
                    a_weight + off * offset1_scalar
                    for a_weight, off in zip(agent_weights, direction1)
                ]
                for offset_idx2 in range(self.grid_size + 1):
                    offset2_scalar = (self.grid_size // 2 - offset_idx2) / (
                        self.grid_size // 2
                    )
                    weights_curr_offsets = [
                        a_weight + off * offset2_scalar
                        for a_weight, off in zip(weights_curr_offset1, direction2)
                    ]
                    weights_offsets[offset_idx1][offset_idx2] = weights_curr_offsets

            weights_offsets_flat = [
                item for sublist in weights_offsets for item in sublist
            ]

            with multiprocessing.get_context("spawn").Pool(self.num_processes) as pool:
                returns_offsets_flat = pool.map(
                    functools.partial(
                        eval_parameters,
                        env_factory=self.env_factory,
                        agent_factory=self.agent_factory,
                        num_steps=self.num_steps,
                    ),
                    weights_offsets_flat,
                )
            returns_offsets = np.array(returns_offsets_flat).reshape(
                self.grid_size + 1, self.grid_size + 1
            )
            data_file = self.data_dir / self._result_filename(env_step, i)
            np.save(str(data_file), returns_offsets)
            coords = np.linspace(-1.0, 1.0, num=self.grid_size + 1)
            x_coords, y_coords = np.meshgrid(coords, coords)
            plot_outpath = self.out_dir / f"{self._result_filename(env_step, i)}.png"
            self._plot_surface(
                x_coords,
                y_coords,
                returns_offsets,
                plot_outpath,
            )
            im = Image.open(plot_outpath)
            im_np = np.array(im)[..., :-1]
            self.summary_writer.add_image(
                f"reward_surfaces/{env_step}_{i}", im_np, dataformats="HWC"
            )

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
        plt.close()

    @staticmethod
    def _result_filename(env_step: int, plot_idx: int) -> str:
        return f"{env_step:07d}_{plot_idx:02d}"
