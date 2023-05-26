from math import sqrt

import pytest
import torch

from action_space_toolbox.analysis.high_curvature_subspace_analysis.high_curvature_subspace_analysis import (
    HighCurvatureSubspaceAnalysis,
)


def test_high_curvature_overlap():
    v1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).T
    overlap = HighCurvatureSubspaceAnalysis._calculate_eigenvectors_overlap(v1, v1)
    assert overlap == pytest.approx(1.0)
    v2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).T
    overlap = HighCurvatureSubspaceAnalysis._calculate_eigenvectors_overlap(v1, v2)
    assert overlap == pytest.approx(0.5)
    v2 = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).T
    overlap = HighCurvatureSubspaceAnalysis._calculate_eigenvectors_overlap(v1, v2)
    assert overlap == pytest.approx(0.0)
    v2 = torch.tensor(
        [[sqrt(0.5), sqrt(0.5), 0.0, 0.0], [0.0, sqrt(0.5), sqrt(0.5), 0.0]]
    ).T
    overlap = HighCurvatureSubspaceAnalysis._calculate_eigenvectors_overlap(v1, v2)
    assert overlap == pytest.approx(0.5 * (1.0 + 0.5))
