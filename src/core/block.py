from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Block:
    """
    Pure geometric block representation.
    """
    vertices: List[Tuple[int, int]]  # polygon vertices (row, col)
    row_range: Tuple[int, int]  # (start, end)
    col_range: Tuple[int, int]  # (start, end)


@dataclass
class BlockStructure:
    """
    Block structure with metadata.
    """
    blocks: List[Block]
    count: int

    A: sp.coo_matrix = None
    row_permutation: Optional[np.ndarray] = None
    col_permutation: Optional[np.ndarray] = None

    # Geometry helpers
    def boundaries(self):
        """
        Rectangular boundaries for plotting.
        """
        for block in self.blocks:
            r0, r1 = block.row_range
            c0, c1 = block.col_range
            yield r0, r1 - r0, c0, c1 - c0
