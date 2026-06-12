from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
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
    # pipstools-specific
    mps_path: Path | None = None,
    k: int | None = None,
    hypergraph: str = 'col',
    hg_objective: str = 'soed',
    var_dense: int | None = 200,
    mpsreader: str = 'highs',
    skip_linking: bool = False,
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
        'spectral', 'sliding_window', 'auto', or 'pipstools'.
    n_blocks : int, optional
        Number of blocks (spectral only; estimated if None).
    min_block_size : int
        Minimum rows per sliding-window block.
    threshold : float
        Density threshold (reserved for future use).
    mps_path : Path, optional
        Path to MPS file (pipstools only).
    k : int, optional
        Number of diagonal blocks to request (pipstools only).
    hypergraph : str
        Hypergraph type: 'col', 'row', 'colrow', 'rowcol' (pipstools only).
    hg_objective : str
        Partitioning objective: 'soed', 'cut', 'km1' (pipstools only).
    var_dense : int or None
        Dense-variable threshold; None disables filtering (pipstools only).
    mpsreader : str
        MPS backend: 'highs' or 'gurobi' (pipstools only).
    skip_linking : bool
        If True, omit coupling-row / linking-col overlays (pipstools only).

    Returns
    -------
    BlockStructure
        Detected blocks.  BlockStructure.A is the matrix the detection was
        run on (spectral permutes it internally; sliding_window does not).
        BlockStructure.metadata holds score data for pipstools results.
    """
    if method == 'pipstools':
        if mps_path is None:
            raise ValueError("mps_path is required for pipstools detection")
        if k is None:
            k = 4
        from src.detection.pipstools_backend import PipstoolsDetection
        result = PipstoolsDetection(
            hypergraph=hypergraph,
            hg_objective=hg_objective,
            var_dense=var_dense,
            mpsreader=mpsreader,
        ).detect(mps_path, k)
        return _detection_result_to_block_structure(result, A, skip_linking=skip_linking)

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


def _detection_result_to_block_structure(
    result,
    A: sp.spmatrix,
    skip_linking: bool = False,
) -> BlockStructure:
    """Convert a pipstools DetectionResult into a BlockStructure for the GUI.

    Partition convention (pipstools):
      partition 1          — linking columns (appear in multiple blocks)
      partitions 2..N+1    — diagonal blocks
      partition N+2 (max)  — coupling rows (linking rows)

    The permuted layout is:
      rows:  [diagonal-block rows in block order] + [coupling rows]
      cols:  [diagonal-block cols in block order] + [linking cols]
    """
    row_part: dict[int, int] = result.row_partition_map
    col_part: dict[int, int] = result.col_partition_map

    link_first = {1}
    row_partition_ids = set(row_part.values())
    col_partition_ids = set(col_part.values())
    link_last = {p for p in row_partition_ids
                 if p not in col_partition_ids and p not in link_first}
    diagonal_ids = sorted(p for p in row_partition_ids
                          if p not in link_first and p not in link_last)

    row_counts = Counter(row_part.values())
    col_counts = Counter(col_part.values())

    # Build row permutation: diagonal-block rows first, then coupling rows
    row_perm: list[int] = []
    for p in diagonal_ids:
        row_perm.extend(sorted(i for i, part in row_part.items() if part == p))
    for p in sorted(link_last):
        row_perm.extend(sorted(i for i, part in row_part.items() if part == p))

    # Build col permutation: diagonal-block cols first, then linking cols
    col_perm: list[int] = []
    for p in diagonal_ids:
        col_perm.extend(sorted(j for j, part in col_part.items() if part == p))
    for p in sorted(link_first):
        col_perm.extend(sorted(j for j, part in col_part.items() if part == p))

    row_perm_arr = np.array(row_perm)
    col_perm_arr = np.array(col_perm)

    A_csr = A.tocsr()
    A_reordered = A_csr[row_perm_arr, :][:, col_perm_arr]

    # Build Block objects for each diagonal block
    blocks: list[Block] = []
    row_offset = 0
    col_offset = 0
    for p in diagonal_ids:
        nr = row_counts[p]
        nc = col_counts.get(p, 0)
        rs, re = row_offset, row_offset + nr
        cs, ce = col_offset, col_offset + nc
        blocks.append(Block(
            vertices=[(rs, cs), (rs, ce), (re, ce), (re, cs)],
            row_range=(rs, re),
            col_range=(cs, ce),
        ))
        row_offset += nr
        col_offset += nc

    n_diag_rows = row_offset
    n_diag_cols = col_offset

    if not skip_linking:
        # Coupling-row strip at the bottom (full diagonal-col width)
        n_coupling = sum(row_counts[p] for p in link_last)
        if n_coupling > 0:
            blocks.append(Block(
                vertices=[(n_diag_rows, 0), (n_diag_rows, n_diag_cols),
                          (n_diag_rows + n_coupling, n_diag_cols),
                          (n_diag_rows + n_coupling, 0)],
                row_range=(n_diag_rows, n_diag_rows + n_coupling),
                col_range=(0, n_diag_cols),
            ))
        # Linking-col strip on the right (full diagonal-row height)
        n_linking = sum(col_counts.get(p, 0) for p in link_first)
        if n_linking > 0:
            blocks.append(Block(
                vertices=[(0, n_diag_cols), (0, n_diag_cols + n_linking),
                          (n_diag_rows, n_diag_cols + n_linking),
                          (n_diag_rows, n_diag_cols)],
                row_range=(0, n_diag_rows),
                col_range=(n_diag_cols, n_diag_cols + n_linking),
            ))

    metadata = {
        'score':         result.score,
        'whitescore':    result.whitescore,
        'n_blocks':      result.n_blocks,
        'coupling_rows': result.coupling_rows,
        'coupling_cols': result.coupling_cols,
    }

    return BlockStructure(
        blocks=blocks,
        A=A_reordered,
        count=len(diagonal_ids),
        row_permutation=row_perm_arr,
        col_permutation=col_perm_arr,
        metadata=metadata,
    )


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
    A_csr = A.tocsr()
    n_rows, n_cols = A_csr.shape

    # Reorderers (e.g. RCM) require a square symmetric matrix. For rectangular
    # constraint matrices, use the row-adjacency A @ A.T for the row ordering.
    if n_rows != n_cols:
        A_sym = (A_csr @ A_csr.T).tocsr()
    else:
        A_sym = A_csr

    row_perm, _ = reorder(A_sym, algorithm)
    col_perm = ut.compute_column_ordering_from_rows(A_csr, row_perm)
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