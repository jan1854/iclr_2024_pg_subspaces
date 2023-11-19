import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def create_plots_iclr_eigenvalues(log_dir, out_dir):
    with (Path(__file__).parent / "res" / "run_configs.yaml").open(
        "r"
    ) as run_configs_file:
        run_configs = yaml.safe_load(run_configs_file)
    for env_name, run_config in run_configs.items():
        if "Finger-spin" not in env_name and "Walker2d" not in env_name:
            continue
        # TODO: Do this for all the run_dirs
        log_path = run_config["log_dirs"]["ppo"]
        ev_cache_path_large = (
            log_dir
            / env_name
            / log_path
            / "0"
            / "cached_results"
            / "eigen"
            / f"eigen_{run_config['xmax']:07d}.npz"
        )
        ev_cache_path_small = (
            log_dir
            / env_name
            / f"{log_path}_bottom_evs"
            / "0"
            / "cached_results"
            / "eigen"
            / f"eigen_{run_config['xmax']:07d}.npz"
        )
        eigen_npz_large = np.load(ev_cache_path_large, allow_pickle=True)
        eigenvals_policy_large = eigen_npz_large["policy.eigenvalues"]
        eigenvals_vf_large = eigen_npz_large["value_function.eigenvalues"]
        # eigen_npz_small = np.load(ev_cache_path_small)
        # eigenvals_policy_small = eigen_npz_small["policy.eigenvalues"]
        # eigenvals_vf_small = eigen_npz_small["value_function.eigenvalues"]
        # missing_evs_policy = (
        #     np.ones(
        #         run_config["policy_size"]
        #         - len(eigenvals_policy_large)
        #         - len(eigenvals_policy_small)
        #     )
        #     * eigenvals_policy_large[-1]
        # )
        # missing_evs_vf = (
        #     np.ones(
        #         run_config["value_function_size"]
        #         - len(eigenvals_vf_large)
        #         - len(eigenvals_vf_small)
        #     )
        #     * eigenvals_vf_large[-1]
        # )
        # assert eigenvals_policy_small[-1] > 0
        # assert eigenvals_vf_small[-1] > 0
        # eigenvals_policy_all = np.concatenate(
        #     (eigenvals_policy_large, missing_evs_policy, eigenvals_policy_small)
        # )
        # eigenvals_vf_all = np.concatenate(
        #     (eigenvals_vf_large, missing_evs_vf, eigenvals_vf_small)
        # )
        eigenvals_policy_all = eigenvals_policy_large
        eigenvals_vf_all = eigenvals_vf_large

        for loss_type, eigenvals in zip(
            ["policy", "value_function"], [eigenvals_policy_all, eigenvals_vf_all]
        ):
            plt.rc("font", size=26)
            fig, ax = plt.subplots()
            bins = 10.0 ** np.arange(-2, 5, 1)
            bins = np.concatenate((-bins, [0], bins))
            bins.sort()
            ax.hist(eigenvals, log=True, bins=bins)
            ax.set_xscale(
                "symlog", linthresh=10 ** (-2)
            )  # Using symlog to handle the 0 in the bins
            ax.set_xticks(bins)  # To display all the bin edges
            ax.set_xlabel("Eigenvalues")
            ax.set_ylabel("Counts")
            ax.set_xlim(min(bins), max(bins))
            ax.set_ylim(0.5, 3000)

            ax.set_xticks([-10000, -100, -1, -1e-2, 1e-2, 1, 100, 10000])
            if env_name == "dmc_Finger-spin_TC-v1":
                env_name_out = "dmc_finger_spin_tc"
            else:
                env_name_out = "gym_walker2d_tc"
            fig.savefig(
                out_dir / f"ppo_{env_name_out}_eigenspectrum_{loss_type}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

            fig, ax = plt.subplots()

            def subsampling_criterion(i):
                return i <= 200 or i >= len(eigenvals) - 200 or i % 10 == 0

            eigenvals_subsampled = np.array(
                [
                    ev
                    for i, ev in enumerate(reversed(eigenvals))
                    if subsampling_criterion(i)
                ]
            )
            ev_indices = np.array(
                [i for i in range(len(eigenvals)) if subsampling_criterion(i)]
            )
            max_abs_ev = np.abs(eigenvals_subsampled).max()
            ax.set_ylim(-max_abs_ev * 1.15, max_abs_ev * 1.15)
            ax.set_xlabel("Index")
            ax.set_ylabel("Eigenvalue")
            ax.scatter(ev_indices, eigenvals_subsampled)
            fig.savefig(
                out_dir / f"ppo_{env_name_out}_eigenspectrum_{loss_type}_plot.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    create_plots_iclr_eigenvalues(Path(args.log_dir), Path(args.out_dir))
