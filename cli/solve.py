"""esp solve — dispatch a model stage to a solver."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser("solve", help="Solve a model stage.")
    p.add_argument("--model", required=True, help="Path to a model folder or .mps file.")
    p.add_argument(
        "--stage",
        default="original",
        help="Which MPS to solve: 'original' or a presolved slug (default: original).",
    )
    p.add_argument(
        "--solver",
        default="pips",
        choices=["pips"],
        help="Solver backend (default: pips).",
    )
    p.add_argument("--np", type=int, default=4, dest="nprocs", help="MPI processes for PIPS (default: 4).")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from src.store.model_store import ModelStore

    model_path = Path(args.model)

    if model_path.suffix == ".mps":
        mps_path = model_path
    else:
        store = ModelStore(model_path.name, root=model_path.parent)
        mps_path = store.original_mps if args.stage == "original" else store.presolved_mps(args.stage)

    if not mps_path.exists():
        print(f"ERROR: input MPS not found: {mps_path}", file=sys.stderr)
        return 1

    if args.solver == "pips":
        return _run_pips(mps_path, args.nprocs)

    print(f"ERROR: unknown solver '{args.solver}'", file=sys.stderr)
    return 1


def _run_pips(mps_path: Path, nprocs: int) -> int:
    import subprocess
    from src.config import PIPS_PATH

    binary = PIPS_PATH / "build" / "PIPS-IPM"
    if not binary.exists():
        print(f"ERROR: PIPS binary not found at {binary}.\n"
              f"Build it with: bash resources/install_pips.sh", file=sys.stderr)
        return 1

    cmd = ["mpirun", "-np", str(nprocs), str(binary), str(mps_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode
