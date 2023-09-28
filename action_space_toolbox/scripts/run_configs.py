RUN_CONFIGS = {
    "Ant_TC-v3": {
        "log_dirs": {"ppo": "2023-09-22/18-15-58", "sac": "2023-09-26/16-54-05"},
        "xmax": 2_000_000,
    },
    "HalfCheetah_TC-v3": {
        "log_dirs": {"ppo": "2023-07-14/21-58-53", "sac": "2023-09-19/11-08-06"},
        "xmax": 3_000_000,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "repeat_after_bugfix"},
    },
    "Pendulum_TC-v1": {
        "log_dirs": {"ppo": "2023-09-22/10-51-27", "sac": "2023-09-22/19-13-25"},
        "xmax": 300_000,
    },
    # "Pendulum_PC-v1": {"log_dirs": {"ppo": "2022-11-18/17-24-43/0"}, "xmax": 300_000},
    # "Pendulum_VC-v1": {"log_dirs": {"ppo": "2022-11-21/19-55-23/0"}, "xmax": 300_000},
    # "Reacher_PC-v2": {"log_dirs": {"ppo": "2022-11-14/13-57-50/0"}, "xmax": 1_000_000},
    # "Reacher_VC-v2": {"log_dirs": {"ppo": "2023-01-13/16-01-01/0"}, "xmax": 1_000_000},
    "Walker2d_TC-v3": {
        "log_dirs": {"ppo": "2023-07-14/23-14-41", "sac": "2023-09-23/23-20-09"},
        "xmax": 2_000_000,
        "policy_size": 5708,
        "value_function_size": 5377,
    },
    "dmc_Cheetah-run_TC-v1": {
        "log_dirs": {"ppo": "2022-11-08/18-05-00"},
        "xmax": 3_000_000,
    },
    "dmc_Ball_in_cup-catch_TC-v1": {
        "log_dirs": {"ppo": "2023-09-22/10-49-35", "sac": "2023-09-24/20-17-17"},
        "xmax": 1_000_000,
    },
    "dmc_Finger-spin_TC-v1": {
        "log_dirs": {"ppo": "2022-12-21/20-44-24", "sac": "2023-09-21/10-34-29"},
        "xmax": 1_000_000,
        "policy_size": 4932,
        "value_function_size": 4865,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "default"},
    },
    "dmc_Walker-walk_TC-v1": {
        "log_dirs": {"ppo": "2022-11-09/17-48-20"},
        "xmax": 3_000_000,
    },
}
