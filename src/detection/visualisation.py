"""Axes-aware partition visualisation, composable for side-by-side plots."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix


def transfer_partition(
    cols_target: pl.DataFrame,
    rows_target: pl.DataFrame,
    cols_source: pl.DataFrame,
    rows_source: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replace partition labels in *target* with values from *source*, joined by name.

    Variables/rows present in target but absent from source (e.g. objcol/objrow
    edge cases) keep their existing target partition as fallback.

    Parameters
    ----------
    cols_target, rows_target : DataFrames with at least [name, partition]
        Typically the presolved model's cols/rows as read from its GDX.
    cols_source, rows_source : DataFrames with at least [name, partition]
        Typically the original model's annotation, used as the label source.

    Returns
    -------
    (cols_updated, rows_updated) with partition column replaced from source.
    """
    def _transfer(target: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
        source_map = source.select("name", "partition").rename({"partition": "_part_src"})
        if "partition" not in target.columns:
            target = target.with_columns(partition=pl.lit(0, dtype=pl.UInt32))
        updated = (
            target
            .join(source_map, on="name", how="left")
            .with_columns(
                partition=pl.when(pl.col("_part_src").is_not_null())
                .then(pl.col("_part_src"))
                .otherwise(pl.col("partition"))
                .cast(pl.UInt32)
            )
            .drop("_part_src")
        )
        return updated

    return _transfer(cols_target, cols_source), _transfer(rows_target, rows_source)


def read_integer_col_names(mps_path) -> set[str]:
    """Return the set of variable names declared as integer/binary in an MPS file."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(mps_path))
    names = set()
    for j in range(h.getNumCol()):
        _, vtype = h.getColIntegrality(j)
        if str(vtype) != "HighsVarType.kContinuous":
            _, name = h.getColName(j)
            names.add(name)
    return names


def plot_partition(
    ax,
    A: pl.DataFrame,
    cols: pl.DataFrame,
    rows: pl.DataFrame,
    colormap: str = "viridis",
    title: str | None = None,
    skip_blocks: bool = False,
    markersize: float = 0.25,
    highlight_row_names: set[str] | None = None,
    highlight_col_names: set[str] | None = None,
    integer_col_names: set[str] | None = None,
) -> None:
    """Render an annotated matrix spy plot into *ax*.

    Replicates pipstools' generate_plot(use_annotation=True) but accepts an
    existing matplotlib Axes so it can be embedded in subplots layouts.

    Parameters
    ----------
    ax : matplotlib Axes
    A : DataFrame with columns [row, col, value] (0-based integer indices)
    cols : DataFrame with columns [name, partition, ...] (one row per variable)
    rows : DataFrame with columns [name, partition, ...] (one row per equation)
    colormap : matplotlib colormap name
    title : axes title
    skip_blocks : if True, show only the first 3 and last 3 diagonal blocks
        plus the linking partitions — useful for large models.
    markersize : spy plot marker size (reduce for large matrices)
    highlight_row_names : row names to mark with dashed horizontal lines after sorting
    highlight_col_names : col names to mark with dashed vertical lines after sorting
    """
    from pipstools.utils import get_blocks_ids, identify_a0_cols
    from pipstools.visualisation import createPatches

    # Optionally restrict to a representative subset of blocks
    if skip_blocks:
        blocks, first, last = get_blocks_ids(rows, cols)
        keep = first + blocks[:3] + blocks[-3:] + last
        rows = rows.with_row_index("row").filter(pl.col("partition").is_in(keep))
        cols = cols.with_row_index("col").filter(pl.col("partition").is_in(keep))
        rename_rows = dict(zip(rows["row"].to_list(), range(len(rows))))
        rename_cols = dict(zip(cols["col"].to_list(), range(len(cols))))
        A = (
            A.join(cols.select("col"), on="col")
            .join(rows.select("row"), on="row")
            .drop_nulls()
            .with_columns(
                row=pl.col("row").replace_strict(rename_rows),
                col=pl.col("col").replace_strict(rename_cols),
            )
        )
        rows = rows.drop("row")
        cols = cols.drop("col")

    # pipstools GDXs sometimes include a trailing objcol not in A
    if len(cols) - len(A["col"].unique()) == 1:
        cols = cols.head(-1)

    # Tag globally-linking columns as partition 0
    cols = identify_a0_cols(A, cols, rows)

    row_partition = rows["partition"].to_numpy()
    col_partition = cols["partition"].to_numpy()

    # Stable sort by partition to produce the block-diagonal layout
    equ_order = row_partition.argsort(kind="mergesort")
    var_order  = col_partition.argsort(kind="mergesort")

    row_partition = row_partition[equ_order]
    col_partition = col_partition[var_order]

    A_coo = coo_matrix(
        (A["value"].to_numpy(), (A["row"].to_numpy(), A["col"].to_numpy()))
    )
    A_csc = A_coo.tocsc()[equ_order, :][:, var_order]

    # Coloured block rectangles
    for patch in createPatches(row_partition, col_partition, colormap=colormap):
        ax.add_patch(patch)

    ax.spy(A_csc, markersize=markersize, precision=0, c="k")
    ax.set_aspect("equal")

    if highlight_row_names and "name" in rows.columns:
        name_to_orig = {name: i for i, name in enumerate(rows["name"].to_list())}
        orig_set = {name_to_orig[n] for n in highlight_row_names if n in name_to_orig}
        for sorted_pos, orig_idx in enumerate(equ_order):
            if orig_idx in orig_set:
                ax.axhline(y=sorted_pos, color="crimson", linewidth=1.0,
                           linestyle="--", alpha=0.65, zorder=5)

    if highlight_col_names and "name" in cols.columns:
        name_to_orig = {name: i for i, name in enumerate(cols["name"].to_list())}
        orig_set = {name_to_orig[n] for n in highlight_col_names if n in name_to_orig}
        for sorted_pos, orig_idx in enumerate(var_order):
            if orig_idx in orig_set:
                ax.axvline(x=sorted_pos, color="crimson", linewidth=1.0,
                           linestyle="--", alpha=0.65, zorder=5)

    if integer_col_names and "name" in cols.columns:
        name_to_orig = {name: i for i, name in enumerate(cols["name"].to_list())}
        orig_set = {name_to_orig[n] for n in integer_col_names if n in name_to_orig}
        for sorted_pos, orig_idx in enumerate(var_order):
            if orig_idx in orig_set:
                ax.axvline(x=sorted_pos, color="limegreen", linewidth=1.2,
                           linestyle="-", alpha=0.8, zorder=6)

    if title:
        ax.set_title(title, fontsize=10)


def compute_partition_stats(
    A: pl.DataFrame,
    cols: pl.DataFrame,
    rows: pl.DataFrame,
    integer_col_names: set[str] | None = None,
) -> dict:
    """Return scoring and coupling stats for an annotated (A, cols, rows) triple.

    Replicates the logic in pipstools' get_score / get_matrix_stats but returns
    a plain dict instead of printing, so it can be rendered however the caller likes.
    """
    from pipstools.utils import get_blocks_ids

    blocks, link_first, link_last = get_blocks_ids(rows, cols)

    col_count = cols.group_by("partition").len().rename({"len": "cols"}).sort("partition")
    row_count = rows.group_by("partition").len().rename({"len": "rows"}).sort("partition")

    # Whitescore
    block_area = (
        row_count.join(col_count, on="partition", how="left")
        .with_columns(pl.col("cols").fill_null(0))
        .with_columns(area=pl.col("rows") * pl.col("cols"))["area"]
        .sum()
    )
    total_area = int(col_count["cols"].sum()) * int(row_count["rows"].sum())
    whitescore = float(1 - block_area / total_area) if total_area else 0.0

    # Linking variables (partition 1 cols that span multiple blocks)
    link_var = 0
    linking_cols_df = (
        cols.with_row_index("col")
        .filter(pl.col("partition").is_in(link_first))
        .select("col")
        .join(A, on="col")
        .select("row", "col")
        .join(rows.with_row_index("row"), on="row")
        .select("col", "partition")
        .group_by("col")
        .n_unique()
    )
    link_var = len(linking_cols_df)

    # A0 rows
    link_equ_a0 = int(
        rows.filter(pl.col("partition").is_in(link_first))["partition"].len()
    )

    # Linking constraint rows
    link_equ_twolink = 0
    link_equ_global  = 0
    linking_rows_F = (
        rows.with_row_index("row")
        .filter(pl.col("partition").is_in(link_last))
        .select("row")
    )
    if len(linking_rows_F) > 0:
        colparts_per_row = (
            cols.select("partition")
            .with_row_index("col")
            .join(A.join(linking_rows_F, on="row"), on="col")
            .group_by("row")
            .agg(pl.col("partition").unique())
        )
        twolinks = (
            colparts_per_row
            .filter(pl.col("partition").list.len() == 2)
            .with_columns(
                (pl.col("partition").list.first().cast(pl.Int64)
                 - pl.col("partition").list.last().cast(pl.Int64)).abs()
            )
            .filter(pl.col("partition") < 2)
        )
        link_equ_twolink = len(twolinks)
        link_equ_global  = len(colparts_per_row) - link_equ_twolink

    est_root_nnz = 2 * (
        link_var ** 2 + link_equ_global ** 2 + link_equ_twolink + link_equ_a0
    )
    score = float(max(0, 100 * whitescore - 100 * (est_root_nnz / 1e8)))

    n_integer_cols = 0
    if integer_col_names and "name" in cols.columns:
        n_integer_cols = int(cols.filter(pl.col("name").is_in(integer_col_names)).height)

    # Block size distribution (diagonal blocks only)
    block_row_counts = (
        row_count.filter(pl.col("partition").is_in(blocks))["rows"]
        if blocks else pl.Series("rows", [], dtype=pl.UInt32)
    )

    # NNZ density per row/col
    row_nnz = A.group_by("row").len()["len"]
    col_nnz = A.group_by("col").len()["len"]

    return {
        "n_rows":             len(rows),
        "n_cols":             len(cols),
        "n_nnz":              len(A),
        "n_integer_cols":     n_integer_cols,
        "n_blocks":           len(blocks),
        "link_cols":          link_var,
        "link_rows_a0":       link_equ_a0,
        "link_rows_twolink":  link_equ_twolink,
        "link_rows_global":   link_equ_global,
        "block_rows_min":     int(block_row_counts.min()) if len(block_row_counts) else 0,
        "block_rows_median":  int(block_row_counts.median()) if len(block_row_counts) else 0,
        "block_rows_max":     int(block_row_counts.max()) if len(block_row_counts) else 0,
        "row_nnz_median":     int(row_nnz.median()) if len(row_nnz) else 0,
        "col_nnz_median":     int(col_nnz.median()) if len(col_nnz) else 0,
        "whitescore":         whitescore,
        "est_root_nnz":       est_root_nnz,
        "score":              score,
    }


def render_stats(ax, stats: dict, title: str | None = None, extra_lines: list[str] | None = None) -> None:
    """Render a compute_partition_stats dict as a text panel into *ax*."""
    ax.axis("off")

    link_rows_total = stats["link_rows_a0"] + stats["link_rows_twolink"] + stats["link_rows_global"]

    lines = [
        f"{'Rows:':<22}{stats['n_rows']:>7}    {'Cols:':<10}{stats['n_cols']:>7}    {'NNZ:':<6}{stats['n_nnz']:>8}    {'Int cols:':<10}{stats.get('n_integer_cols', 0):>5}",
        f"{'Blocks:':<22}{stats['n_blocks']:>7}    {'Whitescore:':<10}{stats['whitescore']:>7.4f}    {'Score:':<6}{stats['score']:>8.4f}",
        f"{'Linking cols:':<22}{stats['link_cols']:>7}    {'Linking rows:':<10}{link_rows_total:>7}  "
        f"(A0={stats['link_rows_a0']}  2-link={stats['link_rows_twolink']}  global={stats['link_rows_global']})",
        f"{'Block rows (min/med/max):':<22}{stats['block_rows_min']:>4} / {stats['block_rows_median']:>4} / {stats['block_rows_max']:>4}"
        f"    {'NNZ/row (med):':<14}{stats['row_nnz_median']:>4}    {'NNZ/col (med):':<14}{stats['col_nnz_median']:>4}",
        f"{'Est. root NNZ:':<22}{stats['est_root_nnz']:>7}",
    ]

    if title:
        lines = [title, ""] + lines
    if extra_lines:
        lines += [""] + extra_lines

    ax.text(
        0.01, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )
