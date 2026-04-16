"""Shared fixtures for detection and reordering tests."""
import numpy as np
import pytest
import scipy.sparse as sp


def _make_connected_square(n: int, density: float = 0.3, seed: int = 0) -> sp.csr_matrix:
    """Return an n×n symmetric sparse matrix guaranteed to be connected.

    A tridiagonal path graph is added as a backbone so that BFS-based
    algorithms (King, Sloan) always have a spanning tree to walk.
    """
    A = sp.random(n, n, density=density, format="csr", random_state=seed)
    A = A + A.T
    path = sp.diags([1.0, 1.0], [-1, 1], shape=(n, n), format="csr")
    A = A + path
    A = (A > 0).astype(float)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


@pytest.fixture
def square_matrix() -> sp.csr_matrix:
    """15×15 connected symmetric sparse matrix."""
    return _make_connected_square(15)


@pytest.fixture
def block_diagonal_square() -> sp.csr_matrix:
    """30×30 block-diagonal matrix — 3 disjoint 10×10 dense blocks."""
    blocks = [
        sp.random(10, 10, density=0.7, format="csr", random_state=i) for i in range(3)
    ]
    return sp.block_diag(blocks, format="csr")


@pytest.fixture
def block_diagonal_rect() -> sp.csr_matrix:
    """24×30 rectangular block-diagonal constraint matrix (3 × 8×10 blocks)."""
    blocks = [
        sp.random(8, 10, density=0.7, format="csr", random_state=i) for i in range(3)
    ]
    return sp.block_diag(blocks, format="csr")
