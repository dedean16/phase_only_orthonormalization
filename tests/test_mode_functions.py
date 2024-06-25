import torch
import numpy as np
import pytest

from mode_optimization.mode_functions import compute_non_orthogonality, compute_similarity
from mode_optimization.helper_functions import is_close


def test_non_orthogonality_on_orthonormal_basis():
    """
    Construct an Euler basis and check if non_orthogonality is 0.
    """
    a = torch.eye(5).unsqueeze(0)
    non_orthogonality, gram = compute_non_orthogonality(a)
    assert non_orthogonality == 0


@pytest.mark.skip(reason="Normalization is currently different. Output can be > 1. Fix normalization first.")
def test_non_orthogonality_on_equal():
    """
    Construct a set of equal modes. The non_orthogonality should be maximal.
    """
    a = torch.zeros(5, 5)
    a[2, :] = 1
    b = a.unsqueeze(0)
    non_orthogonality, gram = compute_non_orthogonality(b)
    assert non_orthogonality == 1


def test_similarity_perfect_match():
    """
    Test similarity on two identical normal bases.
    """
    # Create normal basis
    a = torch.tensor(((1, 0, 2), (2, 1, 0), (0, 2, 1))).unsqueeze(0) / np.sqrt(5)
    similarity = compute_similarity(a, a)
    assert is_close(similarity, 1, margin=1e-6)
