"""Split an annotated MPS problem into per-block GDX files for PIPS-IPM++."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path


def split_to_blocks(
    mps_path: Path,
    target_dir: Path,
    annotation_gdx: Path | None = None,
    k: int | None = None,
    transfer: bool = False,
    hypergraph: str = "row",
    var_dense: int | None = 200,
    mpsreader: str = "highs",
) -> int:
    """Write per-block GDX files into *target_dir* for PIPS-IPM++.

    Three modes, selected by which arguments are provided:

    Direct mode  (``annotation_gdx`` is given, ``transfer=False``):
        Reads the annotated GDX produced by ``annotate_mps`` with
        ``read_gdx``, which returns A, cols, rows and objcoef including the
        objective row already added.  Use this when the annotation was computed
        for the same MPS that is being split.

    Transfer mode  (``annotation_gdx`` is given, ``transfer=True``):
        Reads partition labels from an existing annotated GDX and transfers
        them onto *mps_path* by variable/row name.  Use this when the
        annotation was computed on a different model (e.g. the original model
        being applied to its presolved counterpart).

    Partition mode  (``k`` is given):
        Runs ``get_partitions`` directly on *mps_path* with the requested
        hypergraph type, then writes the distributed blocks.

    Exactly one of *annotation_gdx* or *k* must be supplied.

    Parameters
    ----------
    mps_path : Path
        MPS file passed as ``f_input`` to ``write_gdx``.  In direct mode the
        matrix comes from *annotation_gdx*; in the other modes it is also read
        by ``read_mps``.
    target_dir : Path
        Directory where block GDX files are written.
    annotation_gdx : Path, optional
        Annotated GDX (direct or transfer mode).
    k : int, optional
        Number of diagonal blocks to partition into (partition mode).
    transfer : bool
        When True, treat *annotation_gdx* as a source of partition labels to
        transfer onto *mps_path* (transfer mode).  Ignored when *k* is given.
    hypergraph : str
        Hypergraph type passed to ``get_partitions`` in partition mode.
    var_dense : int or None
        Dense-variable threshold for the hypergraph partitioner.
    mpsreader : str
        MPS reader backend ("highs" or "gurobi").

    Returns
    -------
    int
        Number of diagonal blocks written (N), so PIPS is called with N+1.
    """
    if (annotation_gdx is None) == (k is None):
        raise ValueError("Supply exactly one of annotation_gdx or k.")

    from pipstools.io import write_gdx
    from pipstools.utils import get_blocks_ids

    mps_path = Path(mps_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()

    if annotation_gdx is not None:
        if transfer:
            from pipstools.io import read_mps
            from pipstools.utils import add_objective
            from src.detection.annotation import read_annotation
            from src.detection.visualisation import transfer_partition

            with contextlib.redirect_stdout(buf):
                A, cols, rows, objcoef = read_mps(
                    f_input=mps_path,
                    presolve=False,
                    read_names=True,
                    mpsreader=mpsreader,
                )
            cols_anno, rows_anno = read_annotation(Path(annotation_gdx))
            cols, rows = transfer_partition(
                cols_target=cols,
                rows_target=rows,
                cols_source=cols_anno,
                rows_source=rows_anno,
            )
            unmatched_cols = int((cols["partition"] == 0).sum())
            unmatched_rows = int((rows["partition"] == 0).sum())
            print(
                f"  transfer_partition: {unmatched_cols}/{len(cols)} vars unmatched, "
                f"{unmatched_rows}/{len(rows)} rows unmatched (partition=0)"
            )
            with contextlib.redirect_stdout(buf):
                A, cols, rows = add_objective(A, cols, rows)
        else:
            from pipstools.io import read_gdx

            with contextlib.redirect_stdout(buf):
                A, cols, rows, objcoef = read_gdx(Path(annotation_gdx))
    else:
        from pipstools.io import read_mps
        from pipstools.partitioning import get_partitions
        from pipstools.utils import add_objective

        with contextlib.redirect_stdout(buf):
            A, cols, rows, objcoef = read_mps(
                f_input=mps_path,
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
                hg_objective="soed",
                k=k,
                var_dense=var_dense,
                equ_dense=None,
                regex_pattern=None,
                dec=None,
                vcycles=0,
            )
        print(f"  get_partitions: {k} blocks, hypergraph={hypergraph!r}")
        with contextlib.redirect_stdout(buf):
            A, cols, rows = add_objective(A, cols, rows)

    blocks, _, _ = get_blocks_ids(rows, cols)
    n = len(blocks)

    with contextlib.redirect_stdout(buf):
        write_gdx(
            f_input=mps_path,
            f_output=target_dir / f"block_{n}b",
            A=A,
            cols=cols,
            rows=rows,
            objcoef=objcoef,
            write_names=True,
            write_uels=True,
            add_obj=False,
            distributed=True,
        )

    return n
