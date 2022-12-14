from argparse import ArgumentParser

import numpy as np


def joint_limits(env_name):
    if env_name.startswith("dmc_Cheetah"):
        lim = [[-30, 60], [-50, 50], [-230, 50], [-57, 0.4], [-70, 50], [-28, 28]]
    elif env_name.startswith("dmc_Finger"):
        lim = [[-110, 110], [-110, 110]]
    elif env_name.startswith("dmc_Hopper"):
        lim = [[-30, 30], [-170, 10], [5, 150], [-45, 45]]
    elif env_name.startswith("dmc_Manipulator"):
        # TODO: The hand uses a tendon, which works somewhat different --> Need to handle that somehow
        lim = [[-180, 180], [-160, 160], [-160, 160], [-140, 140], ["?"], ["?"]]
    elif env_name.startswith("dmc_Reacher"):
        lim = [[-180, 180], [-160, 160]]
    elif env_name.startswith("dmc_Walker"):
        lim = [[-20, 100], [-150, 0], [-45, 45], [-20, 100], [-150, 0], [-45, 45]]
    else:
        raise ValueError(f"Unknown environment {env_name}")
    lim = np.array(lim, dtype=float)
    lim[:, 0] += 5
    lim[:, 1] -= 5
    return lim * 2 * np.pi / 360.0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--num-positions", type=int, default=20)
    args = parser.parse_args()

    limits = joint_limits(args.env_name)
    positions = np.array(
        [
            np.random.uniform(limits[:, 0], limits[:, 1])
            for _ in range(args.num_positions)
        ]
    )
    positions = np.around(positions, decimals=2)
    print('"' + args.env_name + '"' + ": " + np.array2string(positions, separator=", "))
