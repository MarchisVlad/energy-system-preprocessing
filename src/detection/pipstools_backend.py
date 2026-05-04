"""Structured wrapper around pipstools hypergraph partitioning.

Returns a DetectionResult instead of printing to stdout.
"""
from __future__ import annotations

import io
import contextlib
from pathlib import Path

import polars as pl

from src.detection.base import DetectionAlgorithm, DetectionResult


class PipstoolsDetection(DetectionAlgorithm):
    """
    Runs pipstools hypergraph partitioning on an MPS file and returns a
    structured DetectionResult.

    Parameters
    ----------
    hypergraph
        Hypergraph type passed to pipstools ('col' or 'row').
    hg_objective
        Objective for partitioning ('soed', 'cut', etc.).
    mpsreader
        MPS reader backend ('highs' or 'gurobi').
    var_dense
        Column density threshold; columns with more entries are treated as linking.
    """

    def __init__(
        self,
        hypergraph: str = "col",
        hg_objective: str = "soed",
        mpsreader: str = "highs",
        var_dense: int | None = 200,
    ):
        self._hypergraph = hypergraph
        self._hg_objective = hg_objective
        self._mpsreader = mpsreader
        self._var_dense = var_dense

    @property
    def name(self) -> str:
        return "pipstools"

    def detect(self, mps_path: Path, k: int) -> DetectionResult:
        """
        Partition the MPS into k blocks and return a DetectionResult.

        Stdout from pipstools is captured and discarded.
        """
        from pipstools.io import read_mps
        from pipstools.partitioning import get_partitions
        from pipstools.utils import get_blocks_ids

        mps_path = Path(mps_path)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            A, cols, rows, _objcoef = read_mps(
                f_input=mps_path,
                presolve=False,
                read_names=True,
                mpsreader=self._mpsreader,
            )
            cols, rows = get_partitions(
                A=A,
                cols=cols,
                rows=rows,
                method="hypergraph",
                hypergraph=self._hypergraph,
                hg_objective=self._hg_objective,
                k=k,
                var_dense=self._var_dense,
                equ_dense=None,
                regex_pattern=None,
                dec=None,
                vcycles=0,
            )

        blocks, link_first, link_last = get_blocks_ids(rows, cols)

        # Row counts per partition
        row_count = (
            rows.with_row_index("idx")
            .group_by("partition")
            .agg(pl.col("idx").count().alias("count"))
        )
        col_count = (
            cols.with_row_index("idx")
            .group_by("partition")
            .agg(pl.col("idx").count().alias("count"))
        )

        partition_to_row_count = dict(
            zip(row_count["partition"].to_list(), row_count["count"].to_list())
        )
        partition_to_col_count = dict(
            zip(col_count["partition"].to_list(), col_count["count"].to_list())
        )

        block_sizes = sorted(
            [partition_to_row_count.get(b, 0) for b in blocks],
            reverse=True,
        )
        coupling_rows = sum(partition_to_row_count.get(p, 0) for p in link_last)
        coupling_cols = sum(partition_to_col_count.get(p, 0) for p in link_first)

        # Partition maps: integer index → partition id
        row_partition_map = {
            i: p
            for i, p in enumerate(rows["partition"].to_list())
        }
        col_partition_map = {
            i: p
            for i, p in enumerate(cols["partition"].to_list())
        }

        whitescore, score = _compute_scores(A, cols, rows, row_count, col_count)

        return DetectionResult(
            mps_path=mps_path,
            k=k,
            n_blocks=len(blocks),
            block_sizes=block_sizes,
            coupling_rows=coupling_rows,
            coupling_cols=coupling_cols,
            row_partition_map=row_partition_map,
            col_partition_map=col_partition_map,
            whitescore=whitescore,
            score=score,
        )


def _compute_scores(
    A: pl.DataFrame,
    cols: pl.DataFrame,
    rows: pl.DataFrame,
    row_count: pl.DataFrame,
    col_count: pl.DataFrame,
) -> tuple[float, float]:
    """Replicate pipstools scoring without printing."""
    from pipstools.utils import get_blocks_ids

    blocks, link_first, link_last = get_blocks_ids(rows, cols)

    col_count_by_part = col_count.rename({"count": "cols"}).sort("partition")
    row_count_by_part = row_count.rename({"count": "rows"}).sort("partition")

    block_area = (
        row_count_by_part.join(col_count_by_part, on="partition", how="left")
        .with_columns(pl.col("cols").fill_null(pl.first("cols")))
        .with_columns(area=pl.col("rows") * pl.col("cols"))["area"]
        .sum()
    )
    total_area = int(col_count_by_part["cols"].sum()) * int(row_count_by_part["rows"].sum())

    whitescore = float(1 - block_area / total_area) if total_area > 0 else 0.0

    # Linking variable count (cols in link_first that span multiple partitions)
    linking_cols_df = (
        cols.with_row_index("col")
        .filter(pl.col("partition").is_in(link_first))
        .select("col")
        .join(A, on="col")
        .select("row", "col")
        .join(cols.with_row_index("row"), on="row")
        .select("col", "partition")
        .group_by("col")
        .n_unique()
    )
    link_var = len(linking_cols_df)

    # Linking equation count
    linking_rows_f = (
        rows.with_row_index("row").filter(pl.col("partition").is_in(link_last)).select("row")
    )
    link_equ_twolink = 0
    link_equ_global = 0
    link_equ_a0 = int(
        len(rows.with_row_index("row").filter(pl.col("partition").is_in(link_first)).select("row"))
    )

    if len(linking_rows_f) > 0:
        colparts_per_row = (
            cols.select("partition")
            .with_row_index("col")
            .join(A.join(linking_rows_f, on="row"), on="col")
            .group_by("row")
            .agg(pl.col("partition").unique())
        )
        twolinks = (
            colparts_per_row.filter(pl.col("partition").list.len() == 2)
            .with_columns(
                (pl.col("partition").list.first() - pl.col("partition").list.last()).abs()
            )
            .filter(pl.col("partition") < 2)
        )
        link_equ_twolink = len(twolinks)
        link_equ_global = len(colparts_per_row) - link_equ_twolink

    est_root_nnz = 2 * (link_var**2 + link_equ_global**2 + link_equ_twolink + link_equ_a0)
    score = float(max(0, 100 * whitescore - 100 * (est_root_nnz / 1e8)))

    return whitescore, score
