"""esp generate — compile a SIMPLE-methods GAMS model to MPS."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "generate", help="Generate an MPS model from a SIMPLE-methods source."
    )
    p.add_argument(
        "--name", required=True, help="Model name (used as the output folder name)."
    )
    p.add_argument(
        "--output", default=None, help="Output root directory (default: data/models/)."
    )
    p.add_argument(
        "--resolution", type=float, default=8, help="RESOLUTION parameter (default: 8)."
    )
    p.add_argument(
        "--from-period",
        type=float,
        default=None,
        dest="from_period",
        help="FROM parameter.",
    )
    p.add_argument(
        "--to-period", type=float, default=None, dest="to_period", help="TO parameter."
    )
    p.add_argument("--regions", type=int, default=None, help="NBREGIONS parameter.")
    p.add_argument(
        "--method", default=None, help="METHOD parameter (SIMPLE model variant)."
    )
    p.add_argument(
        "--simple-root",
        default=None,
        dest="simple_root",
        help="Path to a SIMPLE-methods checkout (default: uses src/config.py).",
    )
    p.add_argument(
        "--integers",
        action="store_true",
        default=False,
        help="Generate a MIP model with integer investment variables (uses simple_mip.gms).",
    )
    p.add_argument(
        "--cap-fraction",
        type=float,
        default=None,
        dest="cap_fraction",
        help=(
            "CAP_FRACTION: scale existing plant capacity by this factor (0 < f ≤ 1). "
            "Values below 1 force capacity expansion. Default: 1.0 (no reduction)."
        ),
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import shutil
    import subprocess
    import sys
    import tempfile

    from src.config import SIMPLE_METHODS_PATH
    from src.store.model_store import ModelStore

    simple_root = Path(args.simple_root) if args.simple_root else SIMPLE_METHODS_PATH
    if not simple_root.exists():
        print(
            f"ERROR: SIMPLE-methods not found at {simple_root}.\n"
            f"Clone it or set --simple-root.",
            file=sys.stderr,
        )
        return 1

    gams_exe = shutil.which("gams")
    if gams_exe is None:
        # Fall back to versioned installs under /data/gams/ (newest first)
        for candidate in sorted(Path("/data/gams").glob("*/gams"), reverse=True):
            if candidate.is_file():
                gams_exe = str(candidate)
                break
    if gams_exe is None:
        print("ERROR: 'gams' not found on PATH or in /data/gams/.", file=sys.stderr)
        return 1

    store = ModelStore(args.name, root=Path(args.output) if args.output else None)

    double_dash = [f"--RESOLUTION={args.resolution}"]
    if args.from_period is not None:
        double_dash.append(f"--FROM={args.from_period}")
    if args.to_period is not None:
        double_dash.append(f"--TO={args.to_period}")
    if args.regions is not None:
        double_dash.append(f"--NBREGIONS={args.regions}")
    if args.method is not None:
        double_dash.append(f"--METHOD={args.method}")
    if args.cap_fraction is not None:
        double_dash.append(f"--CAP_FRACTION={args.cap_fraction}")

    with tempfile.TemporaryDirectory(prefix="esp_generate_") as tmpdir:
        tmp = Path(tmpdir)
        mps_tmp = tmp / "output.mps"

        (tmp / "convert.opt").write_text(f"CplexMPS {mps_tmp.as_posix()}\n")

        gms_file = "simple_mip.gms" if args.integers else "simple.gms"
        solver_arg = "mip=convert" if args.integers else "lp=convert"
        cmd = [gams_exe, gms_file, solver_arg, f"optDir={tmpdir}"] + double_dash
        print(f"Generating '{args.name}': {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=simple_root)

        if result.returncode != 0 or not mps_tmp.is_file():
            print("ERROR: GAMS conversion failed (see listing above).", file=sys.stderr)
            return 1

        store.init_from_mps(mps_tmp)

    print(f"Model saved to {store.original_mps}")
    return 0
