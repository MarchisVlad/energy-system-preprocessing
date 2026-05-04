"""esp compare — diff block structure between two stages of a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "compare",
        help="Compare block structure between two model stages.",
    )
    p.add_argument("--model", required=True, help="Path to a model folder.")
    p.add_argument("--before", required=True, help="Stage slug for the 'before' state (e.g. 'original').")
    p.add_argument("--after", required=True, help="Stage slug for the 'after' state.")
    p.add_argument("--output", default=None, help="Write StructuralDiff JSON to this path.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from src.detection.base import DetectionResult
    from src.detection.compare import compare
    from src.store.model_store import ModelStore

    model_path = Path(args.model)
    store = ModelStore(model_path.name, root=model_path.parent)

    before_dict = store.load_detection(args.before)
    after_dict = store.load_detection(args.after)

    if before_dict is None:
        print(f"ERROR: no detection results for stage '{args.before}'", file=sys.stderr)
        return 1
    if after_dict is None:
        print(f"ERROR: no detection results for stage '{args.after}'", file=sys.stderr)
        return 1

    before = _dict_to_result(before_dict)
    after = _dict_to_result(after_dict)

    diff = compare(before, after)
    print(diff.summary())

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        output_path = store.root / f"comparison_{args.before}_vs_{args.after}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diff.to_dict(), indent=2))
    print(f"\nSaved to: {output_path}")
    return 0


def _dict_to_result(d: dict):
    from src.detection.base import DetectionResult

    return DetectionResult(
        mps_path=Path(d["mps_path"]),
        k=d["k"],
        n_blocks=d["n_blocks"],
        block_sizes=d["block_sizes"],
        coupling_rows=d["coupling_rows"],
        coupling_cols=d["coupling_cols"],
        row_partition_map={},   # not needed for comparison
        col_partition_map={},
        whitescore=d["whitescore"],
        score=d["score"],
    )
