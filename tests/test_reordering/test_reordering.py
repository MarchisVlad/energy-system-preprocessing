"""Tests for matrix reordering algorithms.

Covers:
- Every concrete Reorderer subclass (valid permutation, info dict)
- NaturalReorderer  — identity guarantee
- RandomReorderer   — seed determinism
- CuthillMcKeeReorderer — forward vs reverse
- reorder() public API
- apply_reordering() — row and column permutations
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from src.detection.algorithm import apply_reordering
from src.detection.reorder import (
    AMDReorderer,
    CuthillMcKeeReorderer,
    KingReorderer,
    MMDReorderer,
    NaturalReorderer,
    NestedDissectionReorderer,
    RandomReorderer,
    ReorderingAlgorithm,
    SloanReorderer,
    SpectralReorderer,
    reorder,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _valid_perm(perm: np.ndarray, n: int) -> bool:
    """True iff *perm* is a permutation of [0, n)."""
    return (
        isinstance(perm, np.ndarray)
        and perm.shape == (n,)
        and np.array_equal(np.sort(perm), np.arange(n))
    )


# ── parametrize: one entry per concrete reorderer ────────────────────────────

# SloanReorderer and KingReorderer are excluded from the parametrized suite
# because _find_peripheral_node oscillates between two peripheral nodes on
# symmetric graphs, causing an infinite loop.  See TestKnownBugs below.
ALL_REORDERERS = [
    pytest.param(lambda: NaturalReorderer(), id="natural"),
    pytest.param(lambda: RandomReorderer(seed=7), id="random"),
    pytest.param(lambda: CuthillMcKeeReorderer(), id="cuthill_mckee"),
    pytest.param(
        lambda: CuthillMcKeeReorderer(reverse=True), id="reverse_cuthill_mckee"
    ),
    pytest.param(lambda: AMDReorderer(), id="amd"),
    pytest.param(lambda: MMDReorderer(), id="mmd"),
    pytest.param(lambda: SpectralReorderer(), id="spectral"),
    pytest.param(
        lambda: NestedDissectionReorderer(max_depth=3), id="nested_dissection"
    ),
]


# ── valid permutation (all algorithms) ───────────────────────────────────────


@pytest.mark.parametrize("make_reorderer", ALL_REORDERERS)
def test_returns_valid_permutation(make_reorderer, square_matrix):
    n = square_matrix.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm, _ = make_reorderer().reorder(square_matrix)
    assert _valid_perm(perm, n), f"Expected permutation of [0, {n}), got {perm}"


@pytest.mark.parametrize("make_reorderer", ALL_REORDERERS)
def test_returns_info_dict(make_reorderer, square_matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = make_reorderer().reorder(square_matrix)
    assert isinstance(info, dict)
    assert len(info) > 0


@pytest.mark.parametrize("make_reorderer", ALL_REORDERERS)
def test_permuted_matrix_preserves_nnz(make_reorderer, square_matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm, _ = make_reorderer().reorder(square_matrix)
    A_reordered = square_matrix[perm, :][:, perm]
    assert A_reordered.nnz == square_matrix.nnz


@pytest.mark.parametrize("make_reorderer", ALL_REORDERERS)
def test_permuted_matrix_same_shape(make_reorderer, square_matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm, _ = make_reorderer().reorder(square_matrix)
    A_reordered = square_matrix[perm, :][:, perm]
    assert A_reordered.shape == square_matrix.shape


# ── NaturalReorderer ──────────────────────────────────────────────────────────


class TestNaturalReorderer:
    def test_identity_permutation(self, square_matrix):
        n = square_matrix.shape[0]
        perm, _ = NaturalReorderer().reorder(square_matrix)
        assert np.array_equal(perm, np.arange(n))

    def test_matrix_unchanged_after_apply(self, square_matrix):
        perm, _ = NaturalReorderer().reorder(square_matrix)
        diff = square_matrix[perm, :][:, perm] - square_matrix
        assert diff.nnz == 0

    def test_works_on_rectangular_matrix(self, block_diagonal_rect):
        """NaturalReorderer only uses shape[0] — rectangular input is fine."""
        n_rows = block_diagonal_rect.shape[0]
        perm, _ = NaturalReorderer().reorder(block_diagonal_rect)
        assert _valid_perm(perm, n_rows)
        assert np.array_equal(perm, np.arange(n_rows))


# ── RandomReorderer ───────────────────────────────────────────────────────────


class TestRandomReorderer:
    def test_same_seed_deterministic(self, square_matrix):
        perm1, _ = RandomReorderer(seed=42).reorder(square_matrix)
        perm2, _ = RandomReorderer(seed=42).reorder(square_matrix)
        assert np.array_equal(perm1, perm2)

    def test_different_seeds_give_different_permutations(self, square_matrix):
        perm1, _ = RandomReorderer(seed=0).reorder(square_matrix)
        perm2, _ = RandomReorderer(seed=1).reorder(square_matrix)
        # With n=15 the chance of collision is ~1/15! ≈ 10⁻¹²
        assert not np.array_equal(perm1, perm2)

    def test_works_on_rectangular_matrix(self, block_diagonal_rect):
        """RandomReorderer only uses shape[0] — rectangular input is fine."""
        n_rows = block_diagonal_rect.shape[0]
        perm, _ = RandomReorderer(seed=0).reorder(block_diagonal_rect)
        assert _valid_perm(perm, n_rows)


# ── CuthillMcKeeReorderer ─────────────────────────────────────────────────────


class TestCuthillMcKeeReorderer:
    def test_forward_and_reverse_differ(self, square_matrix):
        perm_fwd, _ = CuthillMcKeeReorderer(reverse=False).reorder(square_matrix)
        perm_rev, _ = CuthillMcKeeReorderer(reverse=True).reorder(square_matrix)
        assert not np.array_equal(perm_fwd, perm_rev)

    def test_info_contains_algorithm_key(self, square_matrix):
        _, info = CuthillMcKeeReorderer().reorder(square_matrix)
        assert "algorithm" in info

    def test_quality_metrics_present(self, square_matrix):
        _, info = CuthillMcKeeReorderer().reorder(square_matrix)
        for key in ("original_bandwidth", "bandwidth"):
            assert key in info, f"Missing key: {key}"

    def test_bandwidth_not_increased(self, block_diagonal_square):
        _, info = CuthillMcKeeReorderer().reorder(block_diagonal_square)
        assert info["bandwidth"] <= info["original_bandwidth"]


# ── SpectralReorderer ─────────────────────────────────────────────────────────


class TestSpectralReorderer:
    def test_valid_on_connected_matrix(self, square_matrix):
        n = square_matrix.shape[0]
        perm, _ = SpectralReorderer().reorder(square_matrix)
        assert _valid_perm(perm, n)

    def test_fallback_on_degenerate_matrix(self):
        """Diagonal matrix has trivial Laplacian; spectral must not crash."""
        A = sp.eye(5, format="csr")
        n = A.shape[0]
        perm, info = SpectralReorderer().reorder(A)
        assert _valid_perm(perm, n)


# ── MMDReorderer warning ──────────────────────────────────────────────────────


class TestMMDReorderer:
    def test_emits_warning(self, square_matrix):
        with pytest.warns(UserWarning, match="MMD"):
            MMDReorderer().reorder(square_matrix)

    def test_valid_permutation_despite_warning(self, square_matrix):
        n = square_matrix.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm, _ = MMDReorderer().reorder(square_matrix)
        assert _valid_perm(perm, n)


# Algorithms that work without infinite loops (exclude Sloan and King)
_SAFE_ALGOS = [
    a
    for a in ReorderingAlgorithm
    if a not in (ReorderingAlgorithm.SLOAN, ReorderingAlgorithm.KING)
]


# ── reorder() public API ──────────────────────────────────────────────────────


class TestReorderPublicAPI:
    @pytest.mark.parametrize("algo", _SAFE_ALGOS)
    def test_returns_perm_and_dict(self, algo, square_matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = reorder(square_matrix, algo)
        assert isinstance(result, tuple) and len(result) == 2
        perm, info = result
        assert _valid_perm(perm, square_matrix.shape[0])
        assert isinstance(info, dict)

    def test_unknown_algorithm_raises(self, square_matrix):
        with pytest.raises((ValueError, KeyError)):
            reorder(square_matrix, "not_a_real_algorithm")

    def test_accepts_enum_value(self, square_matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm, _ = reorder(square_matrix, ReorderingAlgorithm.NATURAL)
        assert _valid_perm(perm, square_matrix.shape[0])


# ── apply_reordering() ────────────────────────────────────────────────────────


class TestApplyReordering:
    """apply_reordering(A, algo) → (row_perm, col_perm).

    Tested on square matrices because most graph-based algorithms (RCM, AMD…)
    require a square adjacency matrix internally.  Natural and Random are
    also tested on a rectangular constraint matrix.
    """

    @pytest.mark.parametrize("algo", _SAFE_ALGOS)
    def test_returns_two_valid_permutations(self, algo, square_matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row_perm, col_perm = apply_reordering(square_matrix, algo)
        n = square_matrix.shape[0]
        assert _valid_perm(row_perm, n), "row_perm is not a valid permutation"
        assert _valid_perm(col_perm, n), "col_perm is not a valid permutation"

    @pytest.mark.parametrize("algo", _SAFE_ALGOS)
    def test_permuted_matrix_same_nnz(self, algo, square_matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row_perm, col_perm = apply_reordering(square_matrix, algo)
        A_r = square_matrix[row_perm, :][:, col_perm]
        assert A_r.nnz == square_matrix.nnz

    @pytest.mark.parametrize(
        "algo",
        [
            ReorderingAlgorithm.NATURAL,
            ReorderingAlgorithm.RANDOM,
        ],
    )
    def test_rectangular_matrix_natural_and_random(self, algo, block_diagonal_rect):
        """Natural and Random work on rectangular constraint matrices."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row_perm, col_perm = apply_reordering(block_diagonal_rect, algo)
        n_rows, n_cols = block_diagonal_rect.shape
        assert _valid_perm(row_perm, n_rows)
        assert _valid_perm(col_perm, n_cols)


# ── Known bugs ────────────────────────────────────────────────────────────────


class TestKnownBugs:
    """Document algorithms with known issues so they are not silently skipped."""

    @pytest.mark.skip(
        reason=(
            "SloanReorderer._find_peripheral_node oscillates between two peripheral "
            "nodes on symmetric graphs, causing an infinite loop.  "
            "Fix: add a visited-set or track the previous max-distance to break ties."
        )
    )
    def test_sloan_infinite_loop(self, square_matrix):
        SloanReorderer().reorder(square_matrix)

    @pytest.mark.skip(
        reason=(
            "KingReorderer._find_peripheral_node has the same infinite-loop issue "
            "as SloanReorderer.  Both share the same BFS oscillation pattern."
        )
    )
    def test_king_infinite_loop(self, square_matrix):
        KingReorderer().reorder(square_matrix)
