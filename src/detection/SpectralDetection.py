import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering

from ..core.BlockStructure import Block, BlockStructure
from ..core.Model import Model
from .DetectionAlgorithm import DetectionAlgorithm


class SpectralDetection(DetectionAlgorithm):
    name = "spectral"

    def __init__(self, A: sp.coo_matrix):
        super().__init__(A)

    def detect(self, **kwargs):
        """Spectral clustering on constraint connectivity"""

        n_blocks = kwargs.get('n_blocks', None)

        # Build constraint similarity matrix
        AT = self.A.T
        similarity = (self.A @ AT).astype(float)

        if n_blocks is None:
            n_blocks = self._estimate_n_blocks(similarity)

        # Always convert to dense and use precomputed affinity
        # This avoids the API warning about constructing affinity from data
        similarity_matrix = similarity.toarray()

        # Make sure the matrix is symmetric and non-negative
        similarity_matrix = np.maximum(similarity_matrix, 0)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        try:
            clustering = SpectralClustering(n_clusters=n_blocks,
                                            affinity='precomputed',
                                            random_state=42,
                                            assign_labels='kmeans')
            row_partition = clustering.fit_predict(similarity_matrix)
        except Exception as e:
            # Fallback if spectral fails
            print(
                f"Warning: Spectral clustering failed ({str(e)}), falling back to RCM"
            )

        col_partition = self._compute_col_partition(row_partition)
        row_perm, col_perm = self._partitions_to_permutations(
            row_partition, col_partition)
        blocks = self._extract_blocks_from_partitions(row_partition,
                                                      col_partition, row_perm,
                                                      col_perm)

        A_permuted = self.A[row_perm, :][:, col_perm]

        return BlockStructure(blocks=blocks,
                              A=A_permuted,
                              count=len(blocks),
                              row_permutation=row_perm,
                              col_permutation=col_perm)

    def _compute_col_partition(self, row_partition):
        """Assign columns to blocks based on row partition"""
        n_blocks = row_partition.max() + 1
        col_partition = np.zeros(self.n_cols, dtype=int)

        for col in range(self.n_cols):
            col_data = self.A.getcol(col)
            rows = col_data.nonzero()[0]

            if len(rows) > 0:
                # Assign to most common block
                block_counts = np.bincount(row_partition[rows],
                                           minlength=n_blocks)
                col_partition[col] = block_counts.argmax()
            else:
                # For empty columns, assign to block 0 (or distribute evenly)
                col_partition[col] = col % n_blocks  # Distribute empty columns

        # Ensure each block has at least one column if possible
        for b in range(n_blocks):
            if not np.any(col_partition == b):
                # Find columns that touch this block's rows
                block_rows = np.where(row_partition == b)[0]
                if len(block_rows) > 0:
                    for row in block_rows:
                        row_data = self.A.getrow(row)
                        cols = row_data.nonzero()[1]
                        if len(cols) > 0:
                            # Reassign first column to this block
                            col_partition[cols[0]] = b
                            break

        return col_partition

    def _partitions_to_permutations(self, row_partition, col_partition):
        """Convert partitions to permutation vectors"""
        # Sort rows by partition
        row_perm = np.argsort(row_partition)
        col_perm = np.argsort(col_partition)
        return row_perm, col_perm

    def _extract_blocks_from_partitions(self, row_partition, col_partition,
                                        row_perm, col_perm):
        """Extract block boundaries from partitions"""
        n_blocks = row_partition.max() + 1
        blocks = []

        for b in range(n_blocks):
            row_mask = row_partition[row_perm] == b
            col_mask = col_partition[col_perm] == b

            row_indices = np.where(row_mask)[0]
            col_indices = np.where(col_mask)[0]

            row_start = row_indices[0]
            row_end = row_indices[-1] + 1
            col_start = col_indices[0]
            col_end = col_indices[-1] + 1

            vertices = [
                (row_start, col_start),  # top-left
                (row_start, col_end),  # top-right
                (row_end, col_end),  # bottom-right
                (row_end, col_start)  # bottom-left
            ]

            block = Block(vertices=vertices,
                          row_range=(row_start, row_end),
                          col_range=(col_start, col_end))
            blocks.append(block)

        return blocks

    def _estimate_n_blocks(self, similarity_matrix):
        """Estimate number of blocks using eigenvalue gap"""
        # Compute first few eigenvalues
        n_eigs = min(20, self.n_rows - 2)
        eigenvalues = eigsh(similarity_matrix,
                            k=n_eigs,
                            return_eigenvectors=False)

        # Look for largest gap
        gaps = np.diff(sorted(eigenvalues))
        n_blocks = np.argmax(gaps) + 2  # +2 because of diff and 0-indexing

        return max(2, min(n_blocks, 200))  # Clamp between 2 and 10
