"""Annotate MPS problems with hypergraph partition labels and persist as GDX."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def annotate_mps(
    mps_path: Path,
    k: int,
    output_gdx: Path,
    hypergraph: str = "col",
    hg_objective: str = "soed",
    var_dense: int | None = 200,
    mpsreader: str = "highs",
    vcycles: int = 0,
    overwrite: bool = False,
    fixed_vertices=None,
    annotation_gdx: Path | None = None,
    mps_path_reduced: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, float | None]:
    """Partition *mps_path* into *k* blocks and write the annotated GDX.

    Returns (cols, rows) DataFrames with columns [name, partition].
    Skips the partitioning and re-reads from disk if the GDX already exists
    and *overwrite* is False.

    Fixed vertices can be supplied in two ways (mutually exclusive):
    - *fixed_vertices*: a dict {name: partition} or DataFrame with [name, partition]
      columns, passed directly to the partitioner.
    - *annotation_gdx* + *mps_path_reduced*: annotation GDX of the original model
      and the reduced (presolved) MPS. Variables/constraints present in the annotation
      but absent from the reduced model (i.e. removed by presolve) are extracted and
      used as fixed vertices. *mps_path* must still point to the original model.
    """
    import contextlib
    import io

    output_gdx = Path(output_gdx)
    if output_gdx.suffix != ".gdx":
        output_gdx = output_gdx.parent / (output_gdx.name + ".gdx")

    if output_gdx.exists() and not overwrite:
        cols, rows = read_annotation(output_gdx)
        return cols, rows, None

    from pipstools.io import read_mps, write_gdx
    from pipstools.partitioning import get_partitions
    from pipstools.scoring import get_score
    from pipstools.utils import add_objective

    output_gdx.parent.mkdir(parents=True, exist_ok=True)

    if (
        fixed_vertices is None
        and annotation_gdx is not None
        and mps_path_reduced is not None
    ):
        cols_ann, rows_ann = read_annotation(annotation_gdx)
        buf_red = io.StringIO()
        with contextlib.redirect_stdout(buf_red):
            _, cols_red, rows_red, _ = read_mps(
                f_input=Path(mps_path_reduced),
                presolve=False,
                read_names=True,
                mpsreader=mpsreader,
            )
        removed_cols = cols_ann.filter(~pl.col("name").is_in(cols_red["name"]))
        removed_rows = rows_ann.filter(~pl.col("name").is_in(rows_red["name"]))
        fixed_vertices = pl.concat([removed_cols, removed_rows])
        print(
            f"Derived fixed vertices from presolve diff: "
            f"{len(removed_cols)} removed variables, {len(removed_rows)} removed constraints "
            f"({len(fixed_vertices)} total)"
        )

    buf = io.StringIO()
    # with contextlib.redirect_stdout(buf):
    A, cols, rows, objcoef = read_mps(
        f_input=Path(mps_path),
        presolve=False,
        read_names=True,
        mpsreader=mpsreader,
    )
    cols, rows = get_partitions(
        A=A,
        cols=cols,
        rows=rows,
        method="hypergraph",
        hypergraph=hypergraph,
        hg_objective=hg_objective,
        k=k,
        var_dense=var_dense,
        equ_dense=None,
        regex_pattern=None,
        dec=None,
        vcycles=vcycles,
        fixed_vertices=fixed_vertices,
    )
    score = get_score(A, cols, rows)
    A_out, cols_out, rows_out = add_objective(A, cols, rows)
    write_gdx(
        f_input=Path(mps_path),
        f_output=output_gdx,
        A=A_out,
        cols=cols_out,
        rows=rows_out,
        objcoef=objcoef,
        write_names=True,
        write_uels=True,
        add_obj=False,
    )

    return (
        cols_out.select("name", "partition"),
        rows_out.select("name", "partition"),
        score,
    )


def read_annotation(gdx_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read name + partition from an annotated GDX without leaking GAMS handles.

    pipstools' read_gdx opens a gdxHandle but never closes it, which leaks one
    handle per call. Here we use Gams2Numpy.gdxReadSymbolRaw directly — it opens
    and closes its own handle per call — and skip reading A and objcoef entirely.
    """
    from gams.core.numpy import Gams2Numpy
    from gamspy_base import directory as gmsdir

    f_gdx = str(Path(gdx_path))
    g2n = Gams2Numpy(gmsdir)

    _, data_i = g2n.gdxReadSymbolRaw(f_gdx, "i")  # row names
    _, data_e = g2n.gdxReadSymbolRaw(
        f_gdx, "e"
    )  # [level, marginal, lower, upper, partition]
    _, data_j = g2n.gdxReadSymbolRaw(f_gdx, "j")  # col names
    _, data_x = g2n.gdxReadSymbolRaw(
        f_gdx, "x"
    )  # [level, marginal, lower, upper, partition]

    rows = pl.DataFrame(
        {
            "name": pl.from_numpy(data_i, schema={"name": pl.String})["name"],
            "partition": pl.from_numpy(
                data_e,
                schema={
                    "level": pl.Float64,
                    "marginal": pl.Float64,
                    "lower": pl.Float64,
                    "upper": pl.Float64,
                    "partition": pl.UInt32,
                },
            )["partition"],
        }
    )
    cols = pl.DataFrame(
        {
            "name": pl.from_numpy(data_j, schema={"name": pl.String})["name"],
            "partition": pl.from_numpy(
                data_x,
                schema={
                    "level": pl.Float64,
                    "marginal": pl.Float64,
                    "lower": pl.Float64,
                    "upper": pl.Float64,
                    "partition": pl.UInt32,
                },
            )["partition"],
        }
    )
    return cols, rows
