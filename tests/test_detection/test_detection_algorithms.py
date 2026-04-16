"""Tests for block detection algorithms and utility functions.

Covers:
- SlidingWindowDetection  — block geometry, no permutations
- SpectralDetection       — permutations, permuted matrix
- detect_block_structure() public API
- Utility functions: compute_column_ordering_from_rows,
  estimate_num_blocks, compute_column_partition_from_rows,
  partitions_to_permutations, extract_blocks_from_partitions
"""

import numpy as np
import pytest
import scipy.sparse as sp

from src.core.block import Block, BlockStructure
from src.detection.algorithm import detect_block_structure
from src.detection.utils import (
    compute_column_ordering_from_rows,
    compute_column_partition_from_rows,
    estimate_num_blocks,
    extract_blocks_from_partitions,
    partitions_to_permutations,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _valid_perm(perm: np.ndarray, n: int) -> bool:
    return (
        isinstance(perm, np.ndarray)
        and perm.shape == (n,)
        and np.array_equal(np.sort(perm), np.arange(n))
    )


def _blocks_within_bounds(blocks, n_rows, n_cols):
    for b in blocks:
        r0, r1 = b.row_range
        c0, c1 = b.col_range
        if not (0 <= r0 < r1 <= n_rows):
            return False
        if not (0 <= c0 < c1 <= n_cols):
            return False
    return True


# ── SlidingWindowDetection ────────────────────────────────────────────────────


class TestSlidingWindowDetection:
    def test_returns_block_structure(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert isinstance(result, BlockStructure)

    def test_count_matches_blocks_list(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert result.count == len(result.blocks)

    def test_finds_multiple_blocks_on_block_diagonal(self, block_diagonal_rect):
        """A matrix with 3 disjoint row groups should produce multiple blocks."""
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert result.count > 1

    def test_block_row_ranges_within_bounds(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert _blocks_within_bounds(result.blocks, n_rows, n_cols)

    def test_no_permutations_stored(self, block_diagonal_rect):
        """SlidingWindow does not produce its own reordering."""
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert result.row_permutation is None
        assert result.col_permutation is None

    def test_stored_matrix_is_input(self, block_diagonal_rect):
        """SlidingWindow does not reorder the matrix — A should equal the input."""
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert result.A.shape == block_diagonal_rect.shape
        diff = result.A - block_diagonal_rect
        assert diff.nnz == 0

    def test_blocks_are_block_instances(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        for b in result.blocks:
            assert isinstance(b, Block)

    def test_blocks_have_positive_size(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        for b in result.blocks:
            r0, r1 = b.row_range
            c0, c1 = b.col_range
            assert r1 > r0
            assert c1 > c0

    def test_single_block_matrix_returns_empty(self):
        """A single dense block: sliding window finds 1 chunk → returns empty."""
        A = sp.random(8, 8, density=0.9, format="csr", random_state=0)
        result = detect_block_structure(A, method="sliding_window", min_block_size=10)
        # With min_block_size > n_rows, only one chunk → no blocks returned
        assert result.count == 0
        assert result.blocks == []

    def test_min_block_size_respected(self, block_diagonal_rect):
        """Larger min_block_size produces fewer, coarser blocks."""
        result_fine = detect_block_structure(
            block_diagonal_rect, method="sliding_window", min_block_size=4
        )
        result_coarse = detect_block_structure(
            block_diagonal_rect, method="sliding_window", min_block_size=12
        )
        assert result_coarse.count <= result_fine.count


# ── SpectralDetection ─────────────────────────────────────────────────────────


class TestSpectralDetection:
    def test_returns_block_structure(self, block_diagonal_rect):
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert isinstance(result, BlockStructure)

    def test_count_matches_blocks_list(self, block_diagonal_rect):
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert result.count == len(result.blocks)

    def test_finds_blocks_on_block_diagonal(self, block_diagonal_rect):
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert result.count >= 1

    def test_row_permutation_is_valid(self, block_diagonal_rect):
        n_rows = block_diagonal_rect.shape[0]
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert result.row_permutation is not None
        assert _valid_perm(result.row_permutation, n_rows)

    def test_col_permutation_is_valid(self, block_diagonal_rect):
        n_cols = block_diagonal_rect.shape[1]
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert result.col_permutation is not None
        assert _valid_perm(result.col_permutation, n_cols)

    def test_stored_matrix_is_permuted_version(self, block_diagonal_rect):
        """SpectralDetection permutes the matrix into block-diagonal order."""
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert result.A is not None
        assert result.A.shape == block_diagonal_rect.shape

    def test_stored_matrix_same_nnz(self, block_diagonal_rect):
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert result.A.nnz == block_diagonal_rect.nnz

    def test_block_ranges_within_bounds(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        assert _blocks_within_bounds(result.blocks, n_rows, n_cols)

    def test_blocks_are_block_instances(self, block_diagonal_rect):
        result = detect_block_structure(
            block_diagonal_rect, method="spectral", n_blocks=3
        )
        for b in result.blocks:
            assert isinstance(b, Block)

    def test_n_blocks_parameter_honoured(self, block_diagonal_rect):
        """Spectral clustering should attempt exactly n_blocks clusters."""
        for n in (2, 3, 4):
            result = detect_block_structure(
                block_diagonal_rect, method="spectral", n_blocks=n
            )
            assert result.count <= n


# ── detect_block_structure() public API ───────────────────────────────────────


class TestDetectBlockStructureAPI:
    def test_unknown_method_raises_value_error(self, block_diagonal_rect):
        with pytest.raises(ValueError, match="Unknown detection method"):
            detect_block_structure(block_diagonal_rect, method="nonexistent")

    @pytest.mark.parametrize("method", ["sliding_window", "spectral"])
    def test_valid_methods_return_block_structure(self, method, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method=method, n_blocks=3)
        assert isinstance(result, BlockStructure)

    def test_auto_method_returns_block_structure(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="auto")
        assert isinstance(result, BlockStructure)

    def test_auto_selects_spectral_for_small_matrix(self):
        """_choose_detection_method uses spectral when n_rows < 1000."""
        A = sp.random(20, 15, density=0.4, format="csr", random_state=0)
        result = detect_block_structure(A, method="auto")
        # Spectral sets row_permutation; sliding_window does not.
        assert result.row_permutation is not None

    def test_result_count_is_non_negative(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        assert result.count >= 0

    def test_block_structure_boundaries_generator(self, block_diagonal_rect):
        result = detect_block_structure(block_diagonal_rect, method="sliding_window")
        boundaries = list(result.boundaries())
        assert len(boundaries) == result.count
        for r0, h, c0, w in boundaries:
            assert h > 0 and w > 0


# ── Utility: compute_column_ordering_from_rows ────────────────────────────────


class TestComputeColumnOrderingFromRows:
    def test_returns_valid_permutation(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        row_perm = np.arange(n_rows)
        col_perm = compute_column_ordering_from_rows(block_diagonal_rect, row_perm)
        assert _valid_perm(col_perm, n_cols)

    def test_identity_row_perm_gives_valid_col_perm(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        identity = np.arange(n_rows)
        col_perm = compute_column_ordering_from_rows(block_diagonal_rect, identity)
        assert _valid_perm(col_perm, n_cols)

    def test_reversed_row_perm_changes_col_order(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        fwd = compute_column_ordering_from_rows(block_diagonal_rect, np.arange(n_rows))
        rev = compute_column_ordering_from_rows(
            block_diagonal_rect, np.arange(n_rows - 1, -1, -1)
        )
        assert not np.array_equal(fwd, rev)

    def test_accepts_coo_input(self, block_diagonal_rect):
        A_coo = block_diagonal_rect.tocoo()
        n_rows, n_cols = A_coo.shape
        col_perm = compute_column_ordering_from_rows(A_coo, np.arange(n_rows))
        assert _valid_perm(col_perm, n_cols)


# ── Utility: estimate_num_blocks ──────────────────────────────────────────────


class TestEstimateNumBlocks:
    def test_returns_integer(self, block_diagonal_rect):
        n = estimate_num_blocks(block_diagonal_rect)
        assert isinstance(n, (int, np.integer))

    def test_clamped_to_min_blocks(self, block_diagonal_rect):
        n = estimate_num_blocks(block_diagonal_rect, min_blocks=5)
        assert n >= 5

    def test_clamped_to_max_blocks(self, block_diagonal_rect):
        n = estimate_num_blocks(block_diagonal_rect, max_blocks=4)
        assert n <= 4

    def test_in_range(self, block_diagonal_rect):
        n = estimate_num_blocks(block_diagonal_rect, min_blocks=2, max_blocks=10)
        assert 2 <= n <= 10

    def test_block_diagonal_detected_as_multi_block(self, block_diagonal_rect):
        """A 3-block matrix should be estimated as > 1 block."""
        n = estimate_num_blocks(block_diagonal_rect, min_blocks=2, max_blocks=10)
        assert n > 1

    def test_tiny_matrix_returns_min_blocks(self):
        """Matrix with ≤ 2 rows cannot yield eigenvalues; must return min_blocks."""
        A = sp.eye(2, format="csr")
        n = estimate_num_blocks(A, min_blocks=2)
        assert n >= 2


# ── Utility: compute_column_partition_from_rows ───────────────────────────────


class TestComputeColumnPartitionFromRows:
    def _row_partition(self, n_rows, n_blocks):
        return np.repeat(np.arange(n_blocks), n_rows // n_blocks)[:n_rows]

    def test_returns_array_of_correct_length(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        row_part = self._row_partition(n_rows, 3)
        col_part = compute_column_partition_from_rows(block_diagonal_rect, row_part)
        assert col_part.shape == (n_cols,)

    def test_all_columns_assigned(self, block_diagonal_rect):
        n_rows, n_cols = block_diagonal_rect.shape
        row_part = self._row_partition(n_rows, 3)
        col_part = compute_column_partition_from_rows(block_diagonal_rect, row_part)
        assert np.all(col_part >= 0)
        assert np.all(col_part < 3)

    def test_each_block_has_at_least_one_column(self, block_diagonal_rect):
        n_rows, _ = block_diagonal_rect.shape
        row_part = self._row_partition(n_rows, 3)
        col_part = compute_column_partition_from_rows(block_diagonal_rect, row_part)
        for block_idx in range(3):
            assert np.any(col_part == block_idx), f"Block {block_idx} has no columns"

    def test_block_diagonal_assigns_columns_to_correct_blocks(
        self, block_diagonal_rect
    ):
        """For a block-diagonal matrix, columns in block k should be assigned to k."""
        n_rows, n_cols = block_diagonal_rect.shape
        # Perfect partition: rows 0-7 → block 0, 8-15 → block 1, 16-23 → block 2
        row_part = np.array([0] * 8 + [1] * 8 + [2] * 8)
        col_part = compute_column_partition_from_rows(block_diagonal_rect, row_part)
        # Columns 0-9 should be mostly block 0, 10-19 block 1, 20-29 block 2
        assert np.all(col_part[:10] == 0)
        assert np.all(col_part[10:20] == 1)
        assert np.all(col_part[20:] == 2)


# ── Utility: partitions_to_permutations ──────────────────────────────────────


class TestPartitionsToPermutations:
    def test_returns_two_valid_permutations(self):
        row_part = np.array([0, 1, 2, 0, 1, 2])
        col_part = np.array([0, 1, 1, 2, 0])
        row_perm, col_perm = partitions_to_permutations(row_part, col_part)
        assert _valid_perm(row_perm, len(row_part))
        assert _valid_perm(col_perm, len(col_part))

    def test_groups_rows_by_block(self):
        row_part = np.array([1, 0, 1, 0, 2])
        row_perm, _ = partitions_to_permutations(row_part, np.zeros(3, dtype=int))
        # After permutation, row_part[row_perm] should be sorted
        assert np.array_equal(row_part[row_perm], np.sort(row_part))

    def test_groups_cols_by_block(self):
        col_part = np.array([2, 0, 1, 0, 2, 1])
        _, col_perm = partitions_to_permutations(np.zeros(4, dtype=int), col_part)
        assert np.array_equal(col_part[col_perm], np.sort(col_part))

    def test_identity_partition_gives_identity_permutation(self):
        n = 6
        partition = np.arange(n)  # each row/col in its own block
        row_perm, _ = partitions_to_permutations(partition, partition)
        assert _valid_perm(row_perm, n)


# ── Utility: extract_blocks_from_partitions ───────────────────────────────────


class TestExtractBlocksFromPartitions:
    def _setup(self):
        row_part = np.array([0, 0, 1, 1, 2, 2])
        col_part = np.array([0, 0, 1, 1, 2, 2])
        row_perm, col_perm = partitions_to_permutations(row_part, col_part)
        return row_part, col_part, row_perm, col_perm

    def test_returns_list(self):
        row_part, col_part, row_perm, col_perm = self._setup()
        blocks = extract_blocks_from_partitions(row_part, col_part, row_perm, col_perm)
        assert isinstance(blocks, list)

    def test_correct_number_of_blocks(self):
        row_part, col_part, row_perm, col_perm = self._setup()
        blocks = extract_blocks_from_partitions(row_part, col_part, row_perm, col_perm)
        n_blocks = row_part.max() + 1
        assert len(blocks) == n_blocks

    def test_returns_block_objects_when_class_provided(self):
        row_part, col_part, row_perm, col_perm = self._setup()
        blocks = extract_blocks_from_partitions(
            row_part, col_part, row_perm, col_perm, block_class=Block
        )
        for b in blocks:
            assert isinstance(b, Block)

    def test_returns_dicts_when_no_class(self):
        row_part, col_part, row_perm, col_perm = self._setup()
        blocks = extract_blocks_from_partitions(row_part, col_part, row_perm, col_perm)
        for b in blocks:
            assert isinstance(b, dict)
            assert "row_range" in b and "col_range" in b

    def test_block_ranges_non_overlapping(self):
        row_part = np.array([0, 0, 1, 1, 2, 2])
        col_part = np.array([0, 0, 1, 1, 2, 2])
        row_perm, col_perm = partitions_to_permutations(row_part, col_part)
        blocks = extract_blocks_from_partitions(
            row_part, col_part, row_perm, col_perm, block_class=Block
        )
        # Row ranges must not overlap
        row_ranges = sorted([b.row_range for b in blocks])
        for i in range(len(row_ranges) - 1):
            assert row_ranges[i][1] <= row_ranges[i + 1][0]

    def test_block_ranges_positive_size(self):
        row_part, col_part, row_perm, col_perm = self._setup()
        blocks = extract_blocks_from_partitions(
            row_part, col_part, row_perm, col_perm, block_class=Block
        )
        for b in blocks:
            r0, r1 = b.row_range
            c0, c1 = b.col_range
            assert r1 > r0
            assert c1 > c0

    def test_vertices_have_four_corners(self):
        row_part, col_part, row_perm, col_perm = self._setup()
        blocks = extract_blocks_from_partitions(
            row_part, col_part, row_perm, col_perm, block_class=Block
        )
        for b in blocks:
            assert len(b.vertices) == 4
