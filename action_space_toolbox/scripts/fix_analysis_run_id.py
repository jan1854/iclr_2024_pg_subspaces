import argparse
import itertools
from pathlib import Path

import yaml

from action_space_toolbox.util.tensorboard_logs import add_new_data_indicator

parser = argparse.ArgumentParser()
parser.add_argument("logpath", type=str)
args = parser.parse_args()

log_dir = Path(args.logpath) / "training"


def maybe_move(src: Path, dest: Path):
    dest.mkdir(exist_ok=True)
    if src.exists():
        src.rename(dest / src.name)


for env_dir in log_dir.iterdir():
    for day_dir in env_dir.iterdir():
        for experiment_dir in day_dir.iterdir():
            for i in itertools.count():
                if (experiment_dir / str(i)).exists():
                    run_dir = experiment_dir / str(i)
                else:
                    break
                analyses_dir = run_dir / "analyses" / "reward_surface_visualization"
                if analyses_dir.exists():
                    maybe_move(analyses_dir / "loss_surface", analyses_dir / "default")
                    maybe_move(
                        analyses_dir / "negative_loss_surface", analyses_dir / "default"
                    )
                    maybe_move(
                        analyses_dir / "reward_surface_discounted",
                        analyses_dir / "default",
                    )
                    maybe_move(
                        analyses_dir / "reward_surface_undiscounted",
                        analyses_dir / "default",
                    )

                if (run_dir / ".analyses.yaml").exists():
                    with (run_dir / ".analyses.yaml").open("r") as analyses_file:
                        analyses_data = yaml.safe_load(analyses_file)
                    if analyses_data is not None:
                        for key, val in analyses_data.items():
                            if isinstance(val, list):
                                analyses_data[key] = {"default": val}
                        with (run_dir / ".analyses.yaml").open("w") as analyses_file:
                            yaml.dump(analyses_data, analyses_file)
                        add_new_data_indicator(run_dir)
