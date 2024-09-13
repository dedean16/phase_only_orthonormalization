import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt

from phase_only_orthonormalization.mode_functions import associated_laguerre_polynomials


### TODO: make it work for 1x5x5 bases
@pytest.mark.skip()
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


@pytest.mark.skip()
def test_similarity_perfect_match():
    """
    Test similarity on two identical normal bases.
    """
    # Create normal basis
    a = torch.tensor(((1, 0, 2), (2, 1, 0), (0, 2, 1))).unsqueeze(0) / np.sqrt(5)
    similarity = compute_similarity(a, a)
    assert torch.allclose(similarity, torch.tensor(1.0))


@pytest.mark.parametrize("a", [0, 1, 2])
def test_associated_laguerre_polynomials(a):
    """
    Test associated Laguerre polynomials.
    """
    do_plot = False

    num_x = 40
    x_min = -2
    x_max = 6
    x = torch.linspace(x_min, x_max, num_x)
    N = 4

    L = torch.zeros(x.numel(), N)
    L[:, 0] = torch.ones(x.shape)
    L[:, 1] = -x + a + 1
    L[:, 2] = (x**2 - 2*(a+2)*x + (a+1)*(a+2)) / 2
    L[:, 3] = (-x**3 + 3*(a+3)*x**2 - 3*(a+2)*(a+3)*x + (a+1)*(a+2)*(a+3)) / 6

    Lass = associated_laguerre_polynomials(x.view(1, -1), a, N, ndim=0).T

    if do_plot:
        plt.plot(L, label=[f'$L_{n}$' for n in range(N)])
        plt.plot(Lass, '--', label=[f'assoc. $L_{n}$' for n in range(N)])
        plt.legend()
        plt.xlabel('x')
        plt.show()

    assert (Lass - L).abs().sum() < 1e-4
