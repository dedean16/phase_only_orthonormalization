import torch
from torch import tensor

from mode_optimization.helper_functions import n_choose_k, factorial


def test_n_choose_k():
    """Test n choose k."""
    assert torch.allclose(n_choose_k(5.0, 4.0), tensor(5.0))


def test_factorial():
    """Test factorial."""
    assert torch.allclose(factorial(5.0), tensor(120.0))
