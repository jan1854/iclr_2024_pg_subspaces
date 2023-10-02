import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from matplotlib import pyplot as plt

import gym
import hydra
from omegaconf import OmegaConf


class Renderer:
    def __init__(self):
        self.img = None

    def render_frame(self, env: gym.Env) -> np.ndarray:
        frame = env.render("rgb_array")
        if self.img is None:
            self.img = plt.imshow(frame)
            plt.axis("off")
        else:
            self.img.set_array(frame)
        plt.pause(0.01)
        plt.draw()
        return frame


def render(path: Path, step: int, num_episodes: int, outpath: Optional[Path]) -> None:
    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    env = gym.make(cfg.env, **cfg.env_args)
    agent = hydra.utils.get_class(cfg.algorithm.algorithm._target_).load(
        path / "checkpoints" / f"{cfg.algorithm.name}_{step}_steps",
        env,
        device=cfg.algorithm.algorithm.device,
    )
    print(f"Evaluating the agent from {path} on {cfg.env}")
    renderer = Renderer()
    frames = []
    for e in range(num_episodes):
        obs = env.reset()
        frames.append(renderer.render_frame(env))
        total_reward = 0
        done = False
        i = 0
        while not done:
            i += 1
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            frames.append(renderer.render_frame(env))
            total_reward += reward
            time.sleep(0.02)
        print(f"Reward: {total_reward}")
    if outpath is not None:
        imageio.mimsave(str(outpath), frames, fps=20)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("step", type=int)
    parser.add_argument("--outpath", type=str)
    parser.add_argument("--num-episodes", type=int, default=10)
    args = parser.parse_args()

    render(Path(args.path), args.step, args.num_episodes, args.outpath)
