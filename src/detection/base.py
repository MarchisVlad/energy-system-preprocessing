from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionResult:
    """Structured output from a block-structure detection run on an MPS file."""

    mps_path: Path
    k: int                              # number of partitions requested

    n_blocks: int                       # diagonal block count
    block_sizes: list[int]              # row count per diagonal block, sorted descending
    coupling_rows: int                  # rows in linking partitions
    coupling_cols: int                  # cols in linking partition

    row_partition_map: dict[int, int]   # row_idx → partition_id
    col_partition_map: dict[int, int]   # col_idx → partition_id

    whitescore: float                   # 1 - block_area / total_area (higher = better)
    score: float                        # composite score from pipstools

    @property
    def n_rows(self) -> int:
        return sum(self.block_sizes) + self.coupling_rows

    @property
    def n_cols(self) -> int:
        return sum(
            count for part, count in
            self._count_cols_per_partition().items()
            if part not in self._link_first_partitions()
        ) + self.coupling_cols

    def _count_cols_per_partition(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, p in self.col_partition_map.items():
            counts[p] = counts.get(p, 0) + 1
        return counts

    def _link_first_partitions(self) -> set[int]:
        # Partition 1 is always the linking-column partition in pipstools
        return {1}

    def to_dict(self) -> dict:
        return {
            "mps_path": str(self.mps_path),
            "k": self.k,
            "n_blocks": self.n_blocks,
            "block_sizes": self.block_sizes,
            "coupling_rows": self.coupling_rows,
            "coupling_cols": self.coupling_cols,
            "whitescore": self.whitescore,
            "score": self.score,
        }


class DetectionAlgorithm(ABC):
    """Base class for block-structure detection algorithms."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self, mps_path: Path, k: int) -> DetectionResult:
        """Detect block structure in *mps_path* requesting *k* blocks."""
        ...
