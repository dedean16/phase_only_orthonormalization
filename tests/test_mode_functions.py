import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt

from phase_only_orthonormalization.mode_functions import associated_laguerre_polynomial, laguerre_gauss_mode
from phase_only_orthonormalization.helper_functions import plot_field


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

    Lass = torch.zeros(x.numel(), N)
    Lass[:, 0] = associated_laguerre_polynomial(x.view(1, -1), a=a, n=0)
    Lass[:, 1] = associated_laguerre_polynomial(x.view(1, -1), a=a, n=1)
    Lass[:, 2] = associated_laguerre_polynomial(x.view(1, -1), a=a, n=2)
    Lass[:, 3] = associated_laguerre_polynomial(x.view(1, -1), a=a, n=3)

    if do_plot:
        plt.plot(L, label=[f'$L_{n}$' for n in range(N)])
        plt.plot(Lass, '--', label=[f'assoc. $L_{n}$' for n in range(N)])
        plt.legend()
        plt.xlabel('x')
        plt.show()

    assert (Lass - L).abs().sum() < 1e-4


def test_laguerre_gauss_mode():
    """
    Test Laguerre Gauss mode
    """
    do_plot = False

    el_max = 2
    p_max = 2
    w0 = 0.3

    num_el = 2*el_max+1
    num_p = p_max+1

    x = torch.linspace(-1.0, 1.0, 100).view(1, -1)
    y = torch.linspace(-1.0, 1.0, 100).view(-1, 1)

    if do_plot:
        plt.figure(figsize=(14, 8))

    count = 1
    for p in range(p_max+1):
        for el in range(-el_max, el_max + 1):
            LG = laguerre_gauss_mode(x, y, el, p, w0)

            if do_plot:
                plt.subplot(num_p, num_el, count)
                scale = 1 / LG.abs().max()
                plot_field(LG, scale=scale)
                plt.xticks([])
                plt.yticks([])

            count += 1

    if do_plot:
        plt.show()
