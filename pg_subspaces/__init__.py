import random
from typing import Tuple

import dm_control.suite
import dmc2gym
import gym

DMC_ENVS = {
    f"dmc_{domain.capitalize()}-{task}-v1": (domain, task)
    for domain, task in dm_control.suite.BENCHMARKING
}


def create_base_env(
    domain: str,
    task: str,
    **kwargs,
):
    # Action repeat is handled by the ActionRepeatWrapper (accessible with the action_repeat argument to gym.make())
    assert "frame_skip" not in kwargs or kwargs["frame_skip"] == 1
    if "seed" not in kwargs:
        kwargs["seed"] = random.randint(0, 2**32 - 1)
    return dmc2gym.make(
        domain,
        task,
        time_limit=1000,
        height=480,
        width=640,
        **kwargs,
    )


for env_name, (domain, task) in DMC_ENVS.items():
    gym.register(
        id=env_name,
        entry_point=create_base_env,
        kwargs={"domain": domain, "task": task},
    )
