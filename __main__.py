from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from src.core.model import Model
from src.core.presolving import Presolver, PresolvingMethod
from src.presolvers.algorithm import presolve as python_presolve
from src.presolvers.papilo_handler import presolve as papilo_presolve
from tools.constraint_matrix_analyser.app import ConstraintMatrixAnalyzer


def start_ui():
    app = QApplication(sys.argv)
    window = ConstraintMatrixAnalyzer()
    window.show()
    sys.exit(app.exec())


def handle_presolve(filename: str, methods: list[str], presolver: Presolver):
    _presolve = papilo_presolve if presolver == Presolver.PaPILO else python_presolve

    model = Model(path=filename)

    for method in methods:

        presolving_method = PresolvingMethod[method]
        print(f"Attempting to presolve {model} using {presolving_method} ({presolver.name}).")

        print(
            f"Matrix has {model.A.shape[0]} rows, {model.A.shape[1]} columns and {model.A.getnnz()} nonzeroes.",
            f"Model has {len(model.model.vars)} variables and {len(model.model.constrs)} constraints. "
        )

        model = _presolve(model=model, method=presolving_method)

        print("PRESOLVE SUCCESSFUL")

        print(
            f"Matrix has {model.A.shape[0]} rows, {model.A.shape[1]} columns and {model.A.getnnz()} nonzeroes.",
            f"Model has {len(model.model.vars)} variables and {len(model.model.constrs)} constraints. "
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("energy-systems")
    subparsers = parser.add_subparsers(dest="command", required=True)

    presolve_cmd = subparsers.add_parser("presolve", help="Run a presolve step.")
    presolve_cmd.add_argument("filename", help="Model file")
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
        help="Presolver implementation to use (default: Static).",
    )
    subparsers.add_parser("app", help="Start the app ui.")

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
