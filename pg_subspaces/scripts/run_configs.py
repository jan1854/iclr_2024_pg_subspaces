RUN_CONFIGS = {
    "Ant-v3": {
        "log_dirs": {"ppo": "2023-09-22/18-15-58", "sac": "2023-09-26/16-54-05"},
        "xmax": 2_000_000,
    },
    "HalfCheetah-v3": {
        "log_dirs": {"ppo": "2023-07-14/21-58-53", "sac": "2023-09-19/11-08-06"},
        "xmax": 3_000_000,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "repeat_after_bugfix"},
    },
    "Pendulum-v1": {
        "log_dirs": {"ppo": "2023-09-22/10-51-27", "sac": "2023-09-22/19-13-25"},
        "xmax": 300_000,
    },
    "Walker2d-v3": {
        "log_dirs": {"ppo": "2023-07-14/23-14-41", "sac": "2023-09-23/23-20-09"},
        "xmax": 2_000_000,
        "policy_size": 5708,
        "value_function_size": 5377,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "default"},
    },
    "dmc_Ball_in_cup-catch-v1": {
        "log_dirs": {"ppo": "2023-09-22/10-49-35", "sac": "2023-09-24/20-17-17"},
        "xmax": 1_000_000,
    },
    "dmc_Finger-spin-v1": {
        "log_dirs": {"ppo": "2022-12-21/20-44-24", "sac": "2023-09-21/10-34-29"},
        "xmax": 1_000_000,
        "policy_size": 4932,
        "value_function_size": 4865,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "default"},
    },
}
