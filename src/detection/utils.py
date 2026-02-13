"""Utility functions for matrix partitioning and block extraction.

This module provides utilities for:
- Computing column orderings based on row permutations
- Estimating optimal number of blocks for matrix partitioning
- Computing column partitions from row partitions
- Converting partitions to permutations
- Extracting block structures from partitioned matrices
"""

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def compute_column_ordering_from_rows(
        matrix: sp.spmatrix, row_permutation: np.ndarray) -> np.ndarray:
    """
    Compute column ordering based on first nonzero row appearance after row reordering.
    
    This function reorders columns such that columns are sorted by the index of their
    first nonzero element in the row-reordered matrix. This is useful for creating
    a more structured sparsity pattern after row reordering.
    
    Parameters
    ----------
    matrix : sp.spmatrix
        Input sparse matrix. Will be converted to COO format if needed.
    row_permutation : np.ndarray
        Permutation vector for rows, where row_permutation[i] = j means
        old row i goes to new row j.
    
    Returns
    -------
    col_permutation : np.ndarray
        Permutation vector for columns, ordered by first nonzero row index.
    
    Notes
    -----
    Columns with no nonzero entries are placed at the end of the ordering.
    
    Examples
    --------
    >>> import scipy.sparse as sp
    >>> A = sp.random(100, 100, density=0.1, format='csr')
    >>> row_perm = np.random.permutation(100)
    >>> col_perm = compute_column_ordering_from_rows(A, row_perm)
    >>> A_reordered = A[row_perm, :][:, col_perm]
    """
    # Convert to COO for efficient column access
    if not sp.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    # Apply row permutation
    matrix_reordered = matrix[row_permutation]

    # Find first nonzero row for each column
    first_nonzero_row = np.full(matrix.shape[1], matrix.shape[0], dtype=int)

    for col_idx in range(matrix.shape[1]):
        col_data = matrix_reordered.getcol(col_idx)
        nonzero_rows = col_data.nonzero()[0]

        if len(nonzero_rows) > 0:
            first_nonzero_row[col_idx] = nonzero_rows[0]

    # Sort columns by first nonzero row index
    col_permutation = np.argsort(first_nonzero_row)

    return col_permutation


def estimate_num_blocks(matrix: sp.spmatrix,
                        min_blocks: int = 2,
                        max_blocks: int = 20,
                        n_eigenvalues: Optional[int] = None) -> int:
    """
    Estimate optimal number of blocks for matrix partitioning using spectral gap.
    
    This function uses the eigenvalue spectrum of the similarity matrix (A @ A.T)
    to estimate the natural number of clusters/blocks in the matrix structure.
    The optimal number is determined by finding the largest gap in sorted eigenvalues,
    which indicates a natural separation in the spectral space.
    
    Parameters
    ----------
    matrix : sp.spmatrix
        Input sparse matrix. Will be converted to COO format if needed.
    min_blocks : int, default=2
        Minimum number of blocks to return.
    max_blocks : int, default=20
        Maximum number of blocks to return.
    n_eigenvalues : int, optional
        Number of eigenvalues to compute. If None, uses min(20, n_rows - 2).
    
    Returns
    -------
    n_blocks : int
        Estimated optimal number of blocks, clamped to [min_blocks, max_blocks].
    
    Notes
    -----
    The algorithm:
    1. Computes similarity matrix S = A @ A.T
    2. Extracts top eigenvalues of S
    3. Finds largest gap in sorted eigenvalues
    4. Number of blocks = gap_position + 2
    5. Clamps result to [min_blocks, max_blocks]
    
    The spectral gap heuristic is based on the observation that distinct clusters
    in the matrix structure often correspond to separated eigenvalue groups.
    
    Examples
    --------
    >>> import scipy.sparse as sp
    >>> A = sp.block_diag([sp.random(50, 50, 0.2) for _ in range(3)])
    >>> n_blocks = estimate_num_blocks(A, min_blocks=2, max_blocks=10)
    >>> print(f"Estimated {n_blocks} blocks")
    
    References
    ----------
    .. [1] Von Luxburg, U. (2007). "A tutorial on spectral clustering."
           Statistics and Computing, 17(4), 395-416.
    """
    if not sp.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    # Compute similarity matrix
    similarity = (matrix @ matrix.T).astype(float)

    # Determine number of eigenvalues to compute
    if n_eigenvalues is None:
        n_eigenvalues = min(20, matrix.shape[0] - 2)
    else:
        n_eigenvalues = min(n_eigenvalues, matrix.shape[0] - 2)

    # Ensure we have enough rows
    if n_eigenvalues < 1:
        return min_blocks

    # Compute eigenvalues
    try:
        eigenvalues = eigsh(similarity,
                            k=n_eigenvalues,
                            return_eigenvectors=False)
    except Exception:
        # Fallback if eigenvalue computation fails
        return min_blocks

    # Find largest gap in sorted eigenvalues
    sorted_eigenvalues = np.sort(eigenvalues)
    gaps = np.diff(sorted_eigenvalues)

    # Number of blocks = position of largest gap + 2
    # (+2 because gap at position k means k+1 eigenvalues before gap)
    n_blocks = np.argmax(gaps) + 2

    # Clamp to valid range
    n_blocks = max(min_blocks, min(n_blocks, max_blocks))

    return n_blocks


def compute_column_partition_from_rows(
        matrix: sp.spmatrix,
        row_partition: np.ndarray,
        default_strategy: str = 'cyclic') -> np.ndarray:
    """
    Compute column partition based on row partition and sparsity pattern.
    
    This function assigns each column to a block based on which block contains
    the most nonzero entries in that column. Empty columns are assigned using
    a fallback strategy. The function also ensures every block has at least one
    column assigned if possible.
    
    Parameters
    ----------
    matrix : sp.spmatrix
        Input sparse matrix. Will be converted to COO format if needed.
    row_partition : np.ndarray
        Array of length n_rows where row_partition[i] indicates which block
        row i belongs to. Block indices should be in range [0, n_blocks-1].
    default_strategy : str, default='cyclic'
        Strategy for assigning empty columns:
        - 'cyclic': Round-robin assignment across blocks
        - 'first': Assign all empty columns to block 0
    
    Returns
    -------
    col_partition : np.ndarray
        Array of length n_cols where col_partition[j] indicates which block
        column j belongs to.
    
    Notes
    -----
    The algorithm:
    1. For each column, count nonzeros in each block
    2. Assign column to block with most nonzeros
    3. Empty columns use fallback strategy
    4. Ensure each block has at least one column by reassigning if needed
    
    Examples
    --------
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>> A = sp.random(100, 80, density=0.1, format='csr')
    >>> row_partition = np.random.randint(0, 5, size=100)
    >>> col_partition = compute_column_partition_from_rows(A, row_partition)
    >>> print(f"Columns distributed across {len(np.unique(col_partition))} blocks")
    """
    if not sp.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    n_rows, n_cols = matrix.shape
    n_blocks = row_partition.max() + 1

    col_partition = np.zeros(n_cols, dtype=int)

    # Assign each column to block with most nonzeros
    for col_idx in range(n_cols):
        col_data = matrix.getcol(col_idx)
        nonzero_rows = col_data.nonzero()[0]

        if len(nonzero_rows) > 0:
            # Count occurrences of each block in this column
            block_counts = np.bincount(row_partition[nonzero_rows],
                                       minlength=n_blocks)
            # Assign to block with maximum count
            col_partition[col_idx] = block_counts.argmax()
        else:
            # Handle empty columns
            if default_strategy == 'cyclic':
                col_partition[col_idx] = col_idx % n_blocks
            else:  # 'first' or default
                col_partition[col_idx] = 0

    # Ensure every block has at least one column
    for block_idx in range(n_blocks):
        if not np.any(col_partition == block_idx):
            # Find rows belonging to this block
            block_rows = np.where(row_partition == block_idx)[0]

            if len(block_rows) > 0:
                # Try to find a column with nonzeros in this block
                for row_idx in block_rows:
                    row_data = matrix.getrow(row_idx)
                    nonzero_cols = row_data.nonzero()[1]

                    if len(nonzero_cols) > 0:
                        # Assign first such column to this block
                        col_partition[nonzero_cols[0]] = block_idx
                        break

    return col_partition


def partitions_to_permutations(
        row_partition: np.ndarray,
        col_partition: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert partition assignments to permutation vectors.
    
    This function converts partition vectors (which assign each element to a block)
    into permutation vectors (which specify a reordering). Elements are ordered
    by their block index, creating contiguous blocks in the permuted matrix.
    
    Parameters
    ----------
    row_partition : np.ndarray
        Array where row_partition[i] indicates which block row i belongs to.
    col_partition : np.ndarray
        Array where col_partition[j] indicates which block column j belongs to.
    
    Returns
    -------
    row_permutation : np.ndarray
        Permutation vector for rows. Rows are ordered by block assignment.
    col_permutation : np.ndarray
        Permutation vector for columns. Columns are ordered by block assignment.
    
    Notes
    -----
    The resulting permutations will group all elements of the same block together.
    Within each block, elements maintain their relative order from the original
    indices.
    
    Examples
    --------
    >>> import numpy as np
    >>> row_partition = np.array([0, 2, 1, 0, 1, 2])
    >>> col_partition = np.array([1, 0, 2, 1, 0])
    >>> row_perm, col_perm = partitions_to_permutations(row_partition, col_partition)
    >>> print("Row permutation:", row_perm)  # Groups: block 0, then 1, then 2
    >>> print("Col permutation:", col_perm)
    
    See Also
    --------
    compute_column_partition_from_rows : Compute column partition from row partition
    extract_blocks_from_partitions : Extract block structures after permutation
    """
    row_permutation = np.argsort(row_partition)
    col_permutation = np.argsort(col_partition)

    return row_permutation, col_permutation


def extract_blocks_from_partitions(row_partition: np.ndarray,
                                   col_partition: np.ndarray,
                                   row_permutation: np.ndarray,
                                   col_permutation: np.ndarray,
                                   block_class=None):
    """
    Extract block structures from partitioned and permuted matrix.
    
    This function identifies the rectangular regions corresponding to each block
    after applying row and column permutations. Each block is defined by its
    row and column ranges in the permuted matrix space.
    
    Parameters
    ----------
    row_partition : np.ndarray
        Array where row_partition[i] indicates which block row i belongs to.
    col_partition : np.ndarray
        Array where col_partition[j] indicates which block column j belongs to.
    row_permutation : np.ndarray
        Permutation vector for rows (typically from partitions_to_permutations).
    col_permutation : np.ndarray
        Permutation vector for columns (typically from partitions_to_permutations).
    block_class : class, optional
        Class to instantiate for each block. Should accept parameters:
        vertices, row_range, col_range. If None, returns dict with block info.
    
    Returns
    -------
    blocks : list
        List of block objects or dictionaries, one per non-empty block.
        Each block contains:
        - vertices: List of 4 corner points [(r0,c0), (r0,c1), (r1,c1), (r1,c0)]
        - row_range: Tuple (row_start, row_end) in permuted space
        - col_range: Tuple (col_start, col_end) in permuted space
    
    Notes
    -----
    Empty blocks (with no rows or columns) are skipped. The vertices define the
    corners of each block in the permuted matrix coordinate system, listed in
    counter-clockwise order starting from top-left.
    
    Examples
    --------
    >>> import numpy as np
    >>> row_partition = np.array([0, 0, 1, 1, 2])
    >>> col_partition = np.array([0, 1, 1, 2])
    >>> row_perm, col_perm = partitions_to_permutations(row_partition, col_partition)
    >>> blocks = extract_blocks_from_partitions(
    ...     row_partition, col_partition, row_perm, col_perm
    ... )
    >>> for i, block in enumerate(blocks):
    ...     print(f"Block {i}: rows {block['row_range']}, cols {block['col_range']}")
    
    See Also
    --------
    partitions_to_permutations : Convert partitions to permutations
    compute_column_partition_from_rows : Compute column partitions
    """
    n_blocks = max(row_partition.max(), col_partition.max()) + 1
    blocks = []

    for block_idx in range(n_blocks):
        # Find row and column indices for this block in permuted space
        row_mask = row_partition[row_permutation] == block_idx
        col_mask = col_partition[col_permutation] == block_idx

        row_indices = np.where(row_mask)[0]
        col_indices = np.where(col_mask)[0]

        # Skip empty blocks
        if len(row_indices) == 0 or len(col_indices) == 0:
            continue

        # Determine block boundaries
        row_start = row_indices[0]
        row_end = row_indices[-1] + 1
        col_start = col_indices[0]
        col_end = col_indices[-1] + 1

        # Define vertices (counter-clockwise from top-left)
        vertices = [
            (row_start, col_start),  # Top-left
            (row_start, col_end),  # Top-right
            (row_end, col_end),  # Bottom-right
            (row_end, col_start)  # Bottom-left
        ]

        # Create block object or dictionary
        if block_class is not None:
            block = block_class(vertices=vertices,
                                row_range=(row_start, row_end),
                                col_range=(col_start, col_end))
        else:
            block = {
                'vertices': vertices,
                'row_range': (row_start, row_end),
                'col_range': (col_start, col_end),
                'block_index': block_idx
            }

        blocks.append(block)

    return blocks


# Convenience function for full pipeline
def partition_matrix_to_blocks(matrix: sp.spmatrix,
                               row_partition: np.ndarray,
                               block_class=None):
    """
    Complete pipeline: partition matrix and extract block structures.
    
    This is a convenience function that combines the full workflow of:
    1. Computing column partition from row partition
    2. Converting partitions to permutations
    3. Extracting block structures
    
    Parameters
    ----------
    matrix : sp.spmatrix
        Input sparse matrix.
    row_partition : np.ndarray
        Partition assignment for rows.
    block_class : class, optional
        Class to instantiate for each block. If None, returns dicts.
    
    Returns
    -------
    blocks : list
        List of block objects or dictionaries.
    row_permutation : np.ndarray
        Row permutation vector.
    col_permutation : np.ndarray
        Column permutation vector.
    col_partition : np.ndarray
        Column partition assignment.
    
    Examples
    --------
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>> A = sp.random(100, 80, density=0.1)
    >>> row_partition = np.random.randint(0, 5, size=100)
    >>> blocks, row_perm, col_perm, col_partition = partition_matrix_to_blocks(A, row_partition)
    >>> A_reordered = A[row_perm, :][:, col_perm]
    """
    # Step 1: Compute column partition
    col_partition = compute_column_partition_from_rows(matrix, row_partition)

    # Step 2: Convert to permutations
    row_permutation, col_permutation = partitions_to_permutations(
        row_partition, col_partition)

    # Step 3: Extract blocks
    blocks = extract_blocks_from_partitions(row_partition,
                                            col_partition,
                                            row_permutation,
                                            col_permutation,
                                            block_class=block_class)

    return blocks, row_permutation, col_permutation, col_partition
