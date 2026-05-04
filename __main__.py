"""Temporary entry point — will be replaced by cli/main.py in the next phase."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.core.model import Model
from src.core.presolving import Presolver, PresolvingMethod
from src.presolvers.registry import resolve
from tools.constraint_matrix_analyser.app import ConstraintMatrixAnalyzer

# Maps PresolvingMethod enum → registry key used by the new interface
_METHOD_KEYS = {
    PresolvingMethod.CoeffTightening: "coeff_strengthening",
    PresolvingMethod.Propagation: "propagation",
    PresolvingMethod.DualFix: "dual_fix",
    PresolvingMethod.FixContinuous: "fix_continuous",
    PresolvingMethod.Sparsify: "sparsify",
    PresolvingMethod.Probing: "probing",
}


def start_ui():
    app = QApplication(sys.argv)
    window = ConstraintMatrixAnalyzer()
    window.show()
    sys.exit(app.exec())


def handle_presolve(filename: str, methods: list[str], presolver: Presolver):
    model = Model(path=filename)

    for method_name in methods:
        method_enum = PresolvingMethod[method_name]
        key = _METHOD_KEYS.get(method_enum, method_name.lower())
        backend_suffix = ":papilo" if presolver == Presolver.PaPILO else ""
        spec = f"{key}{backend_suffix}"
        algo = resolve(spec)

        print(f"Applying {algo.name} to {filename}.")
        print(
            f"Before: {model.A.shape[0]} rows, {model.A.shape[1]} cols, "
            f"{model.A.getnnz()} nnz."
        )

        if presolver == Presolver.PaPILO:
            mps_path = Path(filename)
            result = algo.apply(mps_path, mps_path.parent)
            model = Model(path=str(result.output_mps))
        else:
            algo._run(model)
            model.update_matrix()

        print("PRESOLVE SUCCESSFUL")
        print(
            f"After:  {model.A.shape[0]} rows, {model.A.shape[1]} cols, "
            f"{model.A.getnnz()} nnz."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("energy-systems")
    subparsers = parser.add_subparsers(dest="command", required=True)

    presolve_cmd = subparsers.add_parser("presolve", help="Run a presolve step.")
    presolve_cmd.add_argument("filename", help="Model file (.mps)")
    presolve_cmd.add_argument(
        "-methods",
        nargs="+",
        default=["CoeffTightening"],
        choices=[m.name for m in PresolvingMethod],
        help="Presolve technique(s) to apply in order.",
    )
    presolve_cmd.add_argument(
        "-presolver",
        default="Static",
        choices=[p.name for p in Presolver],
        help="Presolver backend (default: Static).",
    )
    subparsers.add_parser("app", help="Start the PyQt6 UI.")

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    match args.command:
        case "app":
            start_ui()

        case "presolve":
            handle_presolve(args.filename, args.methods, Presolver[args.presolver])


if __name__ == "__main__":
    main()
