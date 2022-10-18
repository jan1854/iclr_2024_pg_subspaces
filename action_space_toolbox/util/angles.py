from typing import Union

import numpy as np


def normalize_angle(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    a = a % (2 * np.pi)
    a = np.where(a > np.pi, a - 2 * np.pi, a)
    return a
