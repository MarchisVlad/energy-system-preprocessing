from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering

from src.core.block import Block, BlockStructure
from src.detection import utils as ut
from src.detection.reorder import ReorderingAlgorithm, reorder


def detect_block_structure(
    A: sp.spmatrix,
    method: str = 'spectral',
    n_blocks: int | None = None,
    min_block_size: int = 10,
    threshold: float = 0.1,
) -> BlockStructure:
    """
    Detect block structure in a matrix.

    Reordering is a separate concern — call apply_reordering() first if you
    want to pre-permute the matrix before detection.

    Parameters
    ----------
    A : sp.spmatrix
        Matrix to analyse (already reordered if desired).
    method : str
        'spectral', 'sliding_window', or 'auto'.
    n_blocks : int, optional
        Number of blocks (spectral only; estimated if None).
    min_block_size : int
        Minimum rows per sliding-window block.
    threshold : float
        Density threshold (reserved for future use).

    Returns
    -------
    BlockStructure
        Detected blocks.  BlockStructure.A is the matrix the detection was
        run on (spectral permutes it internally; sliding_window does not).
    """
    if method == 'auto':
        algorithm = _choose_detection_method(A)
    elif method == 'spectral':
        algorithm = SpectralDetection()
    elif method == 'sliding_window':
        algorithm = SlidingWindowDetection()
    else:
        raise ValueError(f"Unknown detection method: {method!r}")

    return algorithm.detect(A, n_blocks=n_blocks, min_block_size=min_block_size,
                            threshold=threshold)


def apply_reordering(
    A: sp.spmatrix,
    algorithm: ReorderingAlgorithm,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute row and column permutations for the given reordering algorithm.

    This is independent of block detection — you can use it before detection,
    after detection, or on its own.

    Parameters
    ----------
    A : sp.spmatrix
        Matrix to reorder.
    algorithm : ReorderingAlgorithm
        The reordering algorithm to apply.

    Returns
    -------
    row_perm, col_perm : np.ndarray
        Apply to the matrix as A[row_perm, :][:, col_perm].
    """
    row_perm, _ = reorder(A, algorithm)
    col_perm = ut.compute_column_ordering_from_rows(A, row_perm)
    return row_perm, col_perm


class DetectionAlgorithm(ABC):
    name: str

    @abstractmethod
    def detect(self, A: sp.spmatrix, **kwargs) -> BlockStructure:
        pass


def _choose_detection_method(A: sp.spmatrix) -> DetectionAlgorithm:
    """Choose best detection method based on matrix properties."""
    n_rows, _ = A.shape
    if n_rows < 1000:
        return SpectralDetection()
    return SlidingWindowDetection()


class SlidingWindowDetection(DetectionAlgorithm):
    name = "sliding_window"

    def detect(self, A: sp.spmatrix, **kwargs) -> BlockStructure:
        min_block_size = kwargs.get('min_block_size', 10)
        n_rows, _ = A.shape
        block_size = max(min_block_size, min(100, n_rows // 10))

        blocks = []
        i = 0
        while i < n_rows:
            row_end = min(i + block_size, n_rows)
            chunk = A[i:row_end]

            if chunk.nnz > 0:
                cols = chunk.nonzero()[1]
                if len(cols) > 0:
                    blocks.append(
                        Block(
                            vertices=[(i, int(cols.min())), (i, int(cols.max()) + 1),
                                      (row_end, int(cols.max()) + 1), (row_end, int(cols.min()))],
                            row_range=(i, row_end),
                            col_range=(int(cols.min()), int(cols.max()) + 1),
                        )
                    )
            i += block_size

        valid = blocks if len(blocks) > 1 else []
        return BlockStructure(blocks=valid, A=A, count=len(valid))


class SpectralDetection(DetectionAlgorithm):
    name = "spectral"

    def detect(self, A: sp.spmatrix, **kwargs) -> BlockStructure:
        n_blocks = kwargs.get('n_blocks', None)
        if n_blocks is None:
            n_blocks = ut.estimate_num_blocks(A)

        A = A.tocsr()  # CSR required for fancy indexing later
        similarity = (A @ A.T).astype(float)
        similarity_matrix = similarity.toarray()
        similarity_matrix = np.maximum(similarity_matrix, 0)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        clustering = SpectralClustering(
            n_clusters=n_blocks,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans',
        )
        row_partition = clustering.fit_predict(similarity_matrix)
        col_partition = ut.compute_column_partition_from_rows(A, row_partition)

        row_perm, col_perm = ut.partitions_to_permutations(row_partition, col_partition)
        blocks = ut.extract_blocks_from_partitions(
            row_partition, col_partition, row_perm, col_perm, block_class=Block
        )

        return BlockStructure(
            blocks=blocks,
            A=A[row_perm, :][:, col_perm],
            count=len(blocks),
            row_permutation=row_perm,
            col_permutation=col_perm,
        )