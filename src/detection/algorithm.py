from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import issymmetric
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sklearn.cluster import SpectralClustering

from src.core.block import Block, BlockStructure
from src.detection import utils as ut
from src.detection.reorder import ReorderingAlgorithm, reorder


def detect_block_structure(A: sp.spmatrix,
                           method: str | None = 'auto',
                           n_blocks: int | None = 10,
                           min_block_size: int | None = 10,
                           threshold: float | None = 0.1) -> BlockStructure:
    """
    Detect block structure in constraint matrix
    
    Args:
        method: 'metis', 'spectral', 'rcm', 'pattern', or 'auto'
        n_blocks: Number of blocks (if known)
        min_block_size: Minimum rows/cols per block
        threshold: Density threshold for considering blocks separate
    
    Returns:
        BlockStructure object with partition info and reordering
    """
    if method == 'auto':
        algorithm = _choose_detection_method(A)

    if method == 'metis':
        pass
    elif method == 'spectral':
        algorithm = SpectralDetection(A)
    elif method == 'rcm':
        algorithm = RCMDetection(A)
    elif method == 'pattern':
        algorithm = RCMDetection(A)
    else:
        raise ValueError(f"Unknown method: {method}")

    return algorithm.detect(n_blocks=None,
                            min_block_size=min_block_size,
                            threshold=threshold)


class DetectionAlgorithm(ABC):
    name: str

    @abstractmethod
    def __init__(self, reorder_algorithm: Optional[ReorderingAlgorithm] = None):
        self.reorder_algorithm = reorder

    @abstractmethod
    def detect(self, A: sp.coo_matrix, **kwargs) -> BlockStructure:
        pass


def _choose_detection_method(self, A: sp.coo_matrix) -> DetectionAlgorithm:

    n_rows, n_cols = A.shape
    """Choose best detection method based on matrix properties"""
    # For small matrices, use spectral
    if n_rows < 1000:
        return SpectralDetection()
    # For very sparse matrices, try pattern recognition first
    elif A.nnz / (n_rows * n_cols) < 0.01:
        # return PatternDetection(A)
        # TODO: pattern detection
        pass
    # Default to SlidingWindowDetection for medium/large matrices
    else:
        return SlidingWindowDetection(ReorderingAlgorithm.CUTHILL_MCKEE)


class SlidingWindowDetection(DetectionAlgorithm):
    name = "sliding window"

    def __init__(self, reorder_algorithm=None):
        super().__init__(reorder_algorithm)

    def detect(self, A, **kwargs):
        symmetric = (A != A.T).nnz == 0

        if self.reorder_algorithm is not None:
            row_perm = reorder(A, self.reorder_algorithm, symmetric, kwargs)
            col_perm = ut.get_col_ordering(A, row_perm)
        else:
            row_perm = []
            col_perm = []

        blocks = []
        block_size = max(10, min(100, A.n_rows // 10))
        i = 0

        while i < A.n_rows:
            block_rows = slice(i, min(i + block_size, A.n_rows))
            block_data = A[block_rows]

            if block_data.nnz > 0:
                cols_in_block = block_data.nonzero()[1]
                if len(cols_in_block) > 0:
                    col_start = int(cols_in_block.min())
                    col_end = int(cols_in_block.max() + 1)
                    row_start = i
                    row_end = min(i + block_size, A.n_rows)

                    vertices = [(row_start, col_start), (row_start, col_end),
                                (row_end, col_end), (row_end, col_start)]

                    block = Block(vertices=vertices,
                                  row_range=(row_start, row_end),
                                  col_range=(col_start, col_end))
                    blocks.append(block)

            i += block_size

        return BlockStructure(blocks=blocks if len(blocks) > 1 else [],
                              A=A,
                              count=len(blocks) if blocks else 0,
                              row_permutation=row_perm,
                              col_permutation=col_perm)


class SpectralDetection(DetectionAlgorithm):
    name = "spectral"

    def __init__(self, reorder_algorithm=None):
        super().__init__(reorder_algorithm)

    def detect(self, A: sp.spmatrix, **kwargs):
        n_blocks = kwargs.get('n_blocks', None)

        if n_blocks is None:
            n_blocks = ut.estimate_n_blocks(A)

        similarity = (A @ A.T).astype(float)
        similarity_matrix = similarity.toarray()
        similarity_matrix = np.maximum(similarity_matrix, 0)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        clustering = SpectralClustering(n_clusters=n_blocks,
                                        affinity='precomputed',
                                        random_state=42,
                                        assign_labels='kmeans')

        row_partition = clustering.fit_predict(similarity_matrix)
        col_partition = ut.compute_col_partition(self.A, row_partition)

        row_perm, col_perm = ut.partitions_to_permutations(
            row_partition, col_partition)

        blocks = ut.extract_blocks_from_partitions(row_partition, col_partition,
                                                   row_perm, col_perm)

        A_permuted = A[row_perm, :][:, col_perm]

        return BlockStructure(blocks=blocks,
                              A=A_permuted,
                              count=len(blocks),
                              row_permutation=row_perm,
                              col_permutation=col_perm)
