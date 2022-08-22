import numpy as np


class Lowpass:
    def __init__(self, dt: float, cutoff: np.ndarray):
        self.dt = dt
        self.cutoff = cutoff
        self.state = None
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros_like(self.cutoff, dtype=float)

    def filter(self, x_new: np.ndarray) -> np.ndarray:
        assert x_new.shape == self.cutoff.shape, \
            f"Incorrect input shape. Expected: {self.cutoff.shape}, got: {x_new.shape}"
        # Formula: https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
        alpha = self.dt / (self.dt + 1 / (2 * np.pi * self.cutoff))
        y_new = x_new * alpha + (1 - alpha) * self.state
        self.state = y_new
        return y_new
