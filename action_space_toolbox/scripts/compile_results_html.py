from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import yaml


def compile_results_html(log_dir: Path, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    tb_configs_dir = Path(__file__).parent / "res" / "tb_configs"
    for tb_config_path in sorted(tb_configs_dir.iterdir()):
        html_code = ""
        with tb_config_path.open("r") as tb_config_file:
            tb_config = yaml.safe_load(tb_config_file)
        html_code += f"<h1>{tb_config_path.stem}</h1><br>\n"
        html_code += learning_curves_gradient_analysis_plots(
            log_dir, tb_config_path.stem
        )
        for analysis_run_id in [
            "default",
            "gradient_direction_magnitude1",
            "magnitude0.001",
            "gradient_direction_magnitude0.001",
        ]:
            html_code += f"<h2>{analysis_run_id}</h2>"
            for plot_name in ["negative_loss_surface", "reward_surface_discounted"]:
                html_code += f"<h3>{plot_name}</h3>\n"
                html_code += reward_surface_visualization_plots(
                    {
                        name: log_dir / "training" / path
                        for name, path in tb_config.items()
                    },
                    analysis_run_id,
                    plot_name,
                )
        outpath = outdir / f"{tb_config_path.stem}.html"
        with outpath.open("w") as outfile:
            outfile.write(html_code)


def learning_curves_gradient_analysis_plots(log_dir: Path, env_name: str) -> str:
    plot_width = 600
    learning_curves_path = log_dir / "plots" / "learning_curves" / (env_name + ".png")
    gradient_similarity_combined_path = (
        log_dir / "plots" / "gradient_similarity_true_gradient" / f"{env_name}.png"
    )
    gradient_similarity_policy_path = (
        log_dir
        / "plots"
        / "gradient_similarity_true_gradient"
        / f"{env_name}"
        / "value_function_loss.png"
    )
    gradient_similarity_vf_path = (
        log_dir
        / "plots"
        / "gradient_similarity_true_gradient"
        / f"{env_name}"
        / "policy_loss.png"
    )
    html_code = f"<img src={learning_curves_path} width={plot_width}>\n"
    if gradient_similarity_combined_path.exists():
        html_code += (
            f"<img src={gradient_similarity_combined_path} width={plot_width}>\n"
        )
    if gradient_similarity_policy_path.exists():
        html_code += f"<img src={gradient_similarity_policy_path} width={plot_width}>\n"
    if gradient_similarity_vf_path.exists():
        html_code += f"<img src={gradient_similarity_vf_path} width={plot_width}><br>\n"
    return html_code


def reward_surface_visualization_plots(
    tb_config: Dict[str, Path], analysis_run_id: str, plot_name: str
) -> str:
    steps_to_plot = [0, 200_000, 400_000, 1_000_000]
    iframe_height = 400
    iframe_width = 500
    steps_headers = [f"\t\t<th>{step}</th>\n" for step in steps_to_plot]
    html_code_table = (
        f"<table>\n\t<tr>\n\t\t<th></th>\n{''.join(steps_headers)}\t</tr>\n"
    )
    table_empty = True
    for name, path in tb_config.items():
        plot_path = (
            path
            / "0"
            / "analyses"
            / "reward_surface_visualization"
            / analysis_run_id
            / plot_name
        )

        if plot_path.exists():
            plot_paths = []
            for step in steps_to_plot:
                plots_curr_step = list(plot_path.glob(f"*{step:07d}_00.html"))
                if len(plots_curr_step) == 0:
                    plot_paths.append(None)
                else:
                    assert len(plots_curr_step) == 1
                    plot_paths.append(plots_curr_step[0])
            html_code_curr_row = f"\t<tr>\n\t\t<td>{name}</td>\n"
            for plot_path in plot_paths:
                html_code_curr_row += "\t\t<td>"
                if plot_path is None:
                    html_code_curr_row += (
                        f"<iframe src='about:blank' height={iframe_height} "
                        f"width={iframe_width}></iframe>"
                    )
                else:
                    html_code_curr_row += (
                        f"<iframe src='{plot_path.absolute()}' height={iframe_height} "
                        f"width={iframe_width}></iframe>"
                    )
                    table_empty = False
                html_code_curr_row += "</td>\n"
            html_code_curr_row += "\t</tr>\n"
            html_code_table += html_code_curr_row
    html_code_table += "</table>\n"
    if table_empty:
        html_code_table = ""
    return html_code_table


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("outdir", type=str)
    args = parser.parse_args()

    compile_results_html(Path(args.log_dir), Path(args.outdir))
