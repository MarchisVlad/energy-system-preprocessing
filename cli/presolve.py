"""esp presolve — apply a presolving technique to a model.

Method specs:
    coeff_strengthening          Python static implementation
    coeff_strengthening:papilo   PaPILO with only that technique enabled
    papilo                       PaPILO with all default methods

Each invocation creates a new numbered run under data/presolve/runs/{model}/{run}/.
Inside that directory: presolved_001.mps, presolved_002.mps, … and debug logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser("presolve", help="Apply a presolving technique to a model.")
    p.add_argument(
        "--model",
        required=True,
        help="Path to a model folder (must contain original.mps) or a direct .mps file.",
    )
    p.add_argument(
        "--method",
        required=True,
        nargs="+",
        help=(
            "One or more presolving method specs applied in order, e.g. "
            "'coeff_strengthening papilo'. Each step feeds its output into the next."
        ),
    )
    p.add_argument(
        "--stage",
        default="original",
        help="Which MPS to start from: 'original' or a path to any .mps file (default: original).",
    )
    p.add_argument("--output", default=None, help="Override the run output directory.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from src.config import PRESOLVE_RUNS
    from src.presolvers.registry import resolve
    from src.store.model_store import ModelStore

    model_path = Path(args.model)

    # Resolve input MPS and model name.
    if model_path.suffix == ".mps":
        mps_in = model_path
        model_name = model_path.stem
    else:
        store = ModelStore(model_path.name, root=model_path.parent)
        model_name = model_path.name
        if args.stage == "original":
            mps_in = store.original_mps
        else:
            stage_path = Path(args.stage)
            mps_in = stage_path if stage_path.is_absolute() else store.root / args.stage

    if not mps_in.exists():
        print(f"ERROR: input MPS not found: {mps_in}", file=sys.stderr)
        return 1

    # Create a new numbered run directory.
    if args.output:
        run_dir = Path(args.output)
    else:
        runs_root = PRESOLVE_RUNS / model_name
        runs_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            d for d in runs_root.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        next_num = int(existing[-1].name) + 1 if existing else 1
        run_dir = runs_root / f"{next_num:03d}"

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    run_meta: dict = {
        "model": model_name,
        "input": str(mps_in),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    for i, method_spec in enumerate(args.method):
        step = i + 1
        algo = resolve(method_spec)
        output_filename = f"presolved_{step:03d}.mps"
        output_path = run_dir / output_filename

        print(f"[{step}] Applying {algo.name!r} to {mps_in.name} ...")

        result = algo.apply(mps_in, output_path)

        print(f"  Status:   {result.status.value}")
        print(f"  Vars:     {result.n_vars_before} → {result.n_vars_after} "
              f"(Δ {result.vars_reduced:+d})")
        print(f"  Constrs:  {result.n_constraints_before} → {result.n_constraints_after} "
              f"(Δ {result.constraints_reduced:+d})")
        if result.nnz_before >= 0:
            print(f"  NNZ:      {result.nnz_before} → {result.nnz_after} "
                  f"(Δ {result.nnz_reduced:+d})")
        print(f"  Output:   {result.output_mps}")

        if result.debug_log:
            debug_path = run_dir / f"debug_{step:03d}_{algo.slug}.txt"
            debug_path.write_text(result.debug_log)

        run_meta["steps"].append({
            "step": step,
            "method": method_spec,
            "algo": algo.name,
            "status": result.status.value,
            "vars_before": result.n_vars_before,
            "vars_after": result.n_vars_after,
            "constraints_before": result.n_constraints_before,
            "constraints_after": result.n_constraints_after,
            "output_file": output_filename,
        })
        (run_dir / "metadata.json").write_text(json.dumps(run_meta, indent=2))

        mps_in = result.output_mps

    return 0
