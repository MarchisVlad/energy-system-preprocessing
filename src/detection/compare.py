"""Compare detection results and partition annotations."""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

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


# ---------------------------------------------------------------------------
# Name-based partition alignment (original ↔ presolved annotations)
# ---------------------------------------------------------------------------

@dataclass
class PartitionAlignment:
    """Name-joined view of two partition annotations.

    Attributes
    ----------
    cols : DataFrame with columns [name, partition_a, partition_b]
        Variables that survive in both annotations.
    rows : DataFrame with columns [name, partition_a, partition_b]
        Constraints that survive in both annotations.
    eliminated_cols : set[str]
        Variable names present in annotation_a but absent from annotation_b
        (dropped by presolve).
    eliminated_rows : set[str]
        Constraint names present in annotation_a but absent from annotation_b.
    n_cols_a : int
        Total variable count in annotation_a.
    n_rows_a : int
        Total constraint count in annotation_a.
    """

    cols: pl.DataFrame
    rows: pl.DataFrame
    eliminated_cols: set[str]
    eliminated_rows: set[str]
    n_cols_a: int
    n_rows_a: int

    @property
    def n_surviving_cols(self) -> int:
        return len(self.cols)

    @property
    def n_surviving_rows(self) -> int:
        return len(self.rows)

    @property
    def col_survival_rate(self) -> float:
        return self.n_surviving_cols / self.n_cols_a if self.n_cols_a else 0.0

    @property
    def row_survival_rate(self) -> float:
        return self.n_surviving_rows / self.n_rows_a if self.n_rows_a else 0.0

    def to_dict(self) -> dict:
        return {
            "n_cols_a": self.n_cols_a,
            "n_rows_a": self.n_rows_a,
            "n_surviving_cols": self.n_surviving_cols,
            "n_surviving_rows": self.n_surviving_rows,
            "col_survival_rate": self.col_survival_rate,
            "row_survival_rate": self.row_survival_rate,
        }


def align_partitions(
    cols_a: pl.DataFrame,
    rows_a: pl.DataFrame,
    cols_b: pl.DataFrame,
    rows_b: pl.DataFrame,
) -> PartitionAlignment:
    """Join two [name, partition] DataFrames on name.

    *_a is the annotation from the original model; *_b from the presolved model.
    The join is a left-semi from b onto a, so only names surviving in b are kept.
    """
    cols_joined = (
        cols_b.rename({"partition": "partition_b"})
        .join(cols_a.rename({"partition": "partition_a"}), on="name", how="inner")
        .select("name", "partition_a", "partition_b")
    )
    rows_joined = (
        rows_b.rename({"partition": "partition_b"})
        .join(rows_a.rename({"partition": "partition_a"}), on="name", how="inner")
        .select("name", "partition_a", "partition_b")
    )

    names_a_cols = set(cols_a["name"].to_list())
    names_b_cols = set(cols_b["name"].to_list())
    names_a_rows = set(rows_a["name"].to_list())
    names_b_rows = set(rows_b["name"].to_list())

    return PartitionAlignment(
        cols=cols_joined,
        rows=rows_joined,
        eliminated_cols=names_a_cols - names_b_cols,
        eliminated_rows=names_a_rows - names_b_rows,
        n_cols_a=len(cols_a),
        n_rows_a=len(rows_a),
    )


@dataclass
class PartitionSimilarity:
    """Clustering similarity metrics between two partition assignments."""

    col_ari: float
    col_nmi: float
    col_purity: float
    row_ari: float
    row_nmi: float
    row_purity: float

    def to_dict(self) -> dict:
        return {
            "col_ari": self.col_ari,
            "col_nmi": self.col_nmi,
            "col_purity": self.col_purity,
            "row_ari": self.row_ari,
            "row_nmi": self.row_nmi,
            "row_purity": self.row_purity,
        }


def partition_similarity(alignment: PartitionAlignment) -> PartitionSimilarity:
    """Compute ARI, NMI and purity between partition_a and partition_b.

    All metrics are computed on surviving elements only (those present in both
    annotations). Requires sklearn.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import numpy as np

    def _purity(labels_true: list, labels_pred: list) -> float:
        if not labels_true:
            return 0.0
        true = np.array(labels_true)
        pred = np.array(labels_pred)
        total = 0
        for p in np.unique(pred):
            mask = pred == p
            total += np.bincount(true[mask]).max()
        return total / len(true)

    def _metrics(df: pl.DataFrame) -> tuple[float, float, float]:
        if len(df) < 2:
            return 0.0, 0.0, 0.0
        a = df["partition_a"].to_list()
        b = df["partition_b"].to_list()
        ari = float(adjusted_rand_score(a, b))
        nmi = float(normalized_mutual_info_score(a, b, average_method="arithmetic"))
        purity = _purity(a, b)
        return ari, nmi, purity

    col_ari, col_nmi, col_purity = _metrics(alignment.cols)
    row_ari, row_nmi, row_purity = _metrics(alignment.rows)

    return PartitionSimilarity(
        col_ari=col_ari,
        col_nmi=col_nmi,
        col_purity=col_purity,
        row_ari=row_ari,
        row_nmi=row_nmi,
        row_purity=row_purity,
    )
