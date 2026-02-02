from .DetectionAlgorithm import DetectionAlgorithm
from ..core.BlockStructure import BlockStructure
from scipy.sparse.csgraph import reverse_cuthill_mckee


class RCMDetection(DetectionAlgorithm):
    name = "rcm"

    def detect(self, model: Model):
        """Reverse Cuthill-McKee reordering to reveal structure"""

        # Build symmetric structure matrix
        AT = self.A.T
        structure = self.A @ AT
        structure = (structure != 0).astype(int)

        # Get RCM permutation
        try:
            row_perm = reverse_cuthill_mckee(structure, symmetric_mode=True)
        except:
            row_perm = np.arange(self.n_rows)

        col_perm = self._get_col_ordering(row_perm)

        # Detect blocks in reordered matrix
        A_reordered = self.A[row_perm][:, col_perm]
        blocks = self._find_blocks_in_ordered_matrix(A_reordered, row_perm,
                                                     col_perm)

        return BlockStructure(blocks=blocks,
                              row_perm=row_perm,
                              col_perm=col_perm)

    def _find_blocks_in_ordered_matrix(self, A_reordered, row_perm, col_perm):
        """Find diagonal blocks in reordered matrix"""
        # Use a simple sliding window approach
        blocks = []
        block_size = max(10, min(100, self.n_rows // 10))

        i = 0
        while i < self.n_rows:
            # Find extent of nonzeros in this block
            block_rows = slice(i, min(i + block_size, self.n_rows))
            block_data = A_reordered[block_rows]

            if block_data.nnz > 0:
                cols_in_block = block_data.nonzero()[1]
                if len(cols_in_block) > 0:
                    col_start = cols_in_block.min()
                    col_end = cols_in_block.max() + 1

                    blocks.append({
                        'row_start': i,
                        'row_end': min(i + block_size, self.n_rows),
                        'col_start': int(col_start),
                        'col_end': int(col_end),
                        'block_id': len(blocks)
                    })

            i += block_size

        return blocks if len(blocks) > 1 else None
