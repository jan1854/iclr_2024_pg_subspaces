from argparse import ArgumentParser

import gym
import numpy as np
import matplotlib.pyplot as plt

from action_transformations.action_transformation_wrapper import ActionTransformationWrapper
from action_transformations.stateless.tanh_transformation_wrapper import TanhTransformationWrapper


def get_transformation(name: str, env: gym.Env, parameters: np.ndarray):
    if name == "tanh":
        return TanhTransformationWrapper(env, parameters)
    else:
        raise ValueError()


def visualize_action_transformation(action_transformation: ActionTransformationWrapper):
    actions = np.linspace(action_transformation.action_space.low, action_transformation.action_space.high, 100)
    actions_transformed = np.array([action_transformation.transform_action(action) for action in actions])
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], actions_transformed[:, i])
        plt.xlabel("original action")
        plt.ylabel("transformed action")
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("transformation", type=str, choices=["tanh"])
    parser.add_argument("parameters", type=float)
    args = parser.parse_args()

    env = gym.make(args.env)
    transformation = get_transformation(
        args.transformation, env, np.array([args.parameters] * np.ones(env.action_space.shape)))
    visualize_action_transformation(transformation)
