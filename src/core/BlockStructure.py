from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

@dataclass(frozen=True)
class Block:
    """
    Pure geometric block representation.
    """
    vertices: List[Tuple[int, int]]           # polygon vertices (row, col)
    row_range: Tuple[int, int]                # (start, end)
    col_range: Tuple[int, int]                # (start, end)


@dataclass
class BlockStructure:
    """
    Algorithm-agnostic block structure with metadata.
    """

    # ---- Geometry ----
    blocks: List[Block]
    count: int

    row_permutation: Optional[np.ndarray] = None
    col_permutation: Optional[np.ndarray] = None

    # ---- Metadata (NON-STRUCTURAL) ----
    method: Optional[str] = None                    # e.g. "rcm", "spectral"
    detected_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Geometry helpers ----
    def boundaries(self):
        """
        Rectangular boundaries for plotting.
        """
        for block in self.blocks:
            r0, r1 = block.row_range
            c0, c1 = block.col_range
            yield r0, r1 - r0, c0, c1 - c0
