"""Compare two DetectionResult objects to quantify structural change."""
from __future__ import annotations

from dataclasses import dataclass

from src.detection.base import DetectionResult


@dataclass
class StructuralDiff:
    """Difference between two block-structure detection results."""

    before: DetectionResult
    after: DetectionResult

    @property
    def n_blocks_delta(self) -> int:
        return self.after.n_blocks - self.before.n_blocks

    @property
    def coupling_rows_delta(self) -> int:
        return self.after.coupling_rows - self.before.coupling_rows

    @property
    def coupling_cols_delta(self) -> int:
        return self.after.coupling_cols - self.before.coupling_cols

    @property
    def whitescore_delta(self) -> float:
        return self.after.whitescore - self.before.whitescore

    @property
    def score_delta(self) -> float:
        return self.after.score - self.before.score

    @property
    def block_sizes_changed(self) -> bool:
        return sorted(self.before.block_sizes) != sorted(self.after.block_sizes)

    def summary(self) -> str:
        lines = [
            f"Blocks:        {self.before.n_blocks} → {self.after.n_blocks} "
            f"({self.n_blocks_delta:+d})",
            f"Coupling rows: {self.before.coupling_rows} → {self.after.coupling_rows} "
            f"({self.coupling_rows_delta:+d})",
            f"Coupling cols: {self.before.coupling_cols} → {self.after.coupling_cols} "
            f"({self.coupling_cols_delta:+d})",
            f"Whitescore:    {self.before.whitescore:.5f} → {self.after.whitescore:.5f} "
            f"({self.whitescore_delta:+.5f})",
            f"Score:         {self.before.score:.5f} → {self.after.score:.5f} "
            f"({self.score_delta:+.5f})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "n_blocks_delta": self.n_blocks_delta,
            "coupling_rows_delta": self.coupling_rows_delta,
            "coupling_cols_delta": self.coupling_cols_delta,
            "whitescore_delta": self.whitescore_delta,
            "score_delta": self.score_delta,
            "block_sizes_changed": self.block_sizes_changed,
        }


def compare(before: DetectionResult, after: DetectionResult) -> StructuralDiff:
    """Return a StructuralDiff between two detection results."""
    return StructuralDiff(before=before, after=after)
