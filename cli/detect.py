"""esp detect — run pipstools block-structure detection on a model stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser("detect", help="Detect block structure of a model stage.")
    p.add_argument(
        "--model",
        required=True,
        help="Path to a model folder or a direct .mps file.",
    )
    p.add_argument(
        "--stage",
        default="original",
        help="Which MPS to analyse: 'original' or a presolved stage slug (default: original).",
    )
    p.add_argument(
        "-k", "--blocks",
        type=int,
        required=True,
        dest="k",
        help="Number of blocks to partition into.",
    )
    p.add_argument(
        "--mpsreader",
        default="highs",
        choices=["highs", "gurobi"],
        help="MPS reader backend (default: highs).",
    )
    p.add_argument("--output", default=None, help="Output directory override.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from src.detection.pipstools_backend import PipstoolsDetection
    from src.store.model_store import ModelStore

    model_path = Path(args.model)

    if model_path.suffix == ".mps":
        mps_path = model_path
        store = None
        stage = args.stage
    else:
        store = ModelStore(model_path.name, root=model_path.parent)
        stage = args.stage
        if stage == "original":
            mps_path = store.original_mps
        else:
            mps_path = store.presolved_mps(stage)

    if not mps_path.exists():
        print(f"ERROR: input MPS not found: {mps_path}", file=sys.stderr)
        return 1

    detector = PipstoolsDetection(mpsreader=args.mpsreader)
    print(f"Detecting block structure of {mps_path.name} (k={args.k}) ...")

    result = detector.detect(mps_path, k=args.k)

    print(f"Blocks:        {result.n_blocks}")
    print(f"Block sizes:   {result.block_sizes}")
    print(f"Coupling rows: {result.coupling_rows}")
    print(f"Coupling cols: {result.coupling_cols}")
    print(f"Whitescore:    {result.whitescore:.5f}")
    print(f"Score:         {result.score:.5f}")

    if store is not None:
        out_path = store.save_detection(stage, result.to_dict())
        print(f"Saved to:      {out_path}")
    elif args.output:
        out = Path(args.output) / f"detection_{stage}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Saved to:      {out}")

    return 0
