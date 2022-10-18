import numpy as np
import pytest

from action_space_toolbox.util.angles import normalize_angle


def test_angle_normalization():
    assert normalize_angle(np.pi) == pytest.approx(np.pi)
    assert normalize_angle(-np.pi) == pytest.approx(np.pi)
    assert normalize_angle(0.0) == pytest.approx(0.0)
    assert normalize_angle(np.pi + 0.05) == pytest.approx(-np.pi + 0.05)
    assert normalize_angle(-np.pi - 0.05) == pytest.approx(np.pi - 0.05)
    assert normalize_angle(4 * np.pi) == pytest.approx(0.0)
    assert normalize_angle(3 * np.pi) == pytest.approx(np.pi)
    assert normalize_angle(
        np.array([0.5 * np.pi, 1.2 * np.pi, -1.3 * np.pi])
    ) == pytest.approx(np.array([0.5 * np.pi, -0.8 * np.pi, 0.7 * np.pi]))
