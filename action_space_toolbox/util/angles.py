import numpy as np


def normalize_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi / 2) % np.pi - np.pi / 2
