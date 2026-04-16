from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from src.core.model import Model
from src.core.presolving import PresolvingMethod
from src.presolvers.algorithm import presolve
from tools.constraint_matrix_analyser.app import ConstraintMatrixAnalyzer


def start_ui():
    app = QApplication(sys.argv)
    window = ConstraintMatrixAnalyzer()
    window.show()
    sys.exit(app.exec())


def handle_presolve(filename: str, methods: list[str]):
    model = Model(path=filename)

    for method in methods:

        presolving_method = PresolvingMethod[method]
        print(f"Attempting to presolve {model} using {presolving_method}.")

        print(
            f"Matrix has {model.A.shape[0]} rows, {model.A.shape[1]} columns and {model.A.getnnz()} nonzeroes."
        )

        model = presolve(model=model, method=presolving_method)

        print("PRESOLVE SUCCESSFUL")

        print(
            f"Matrix has {model.A.shape[0]} rows, {model.A.shape[1]} columns and {model.A.getnnz()} nonzeroes."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("energy-systems")
    subparsers = parser.add_subparsers(dest="command", required=True)

    presolve = subparsers.add_parser("presolve", help="Run a presolve step.")
    presolve.add_argument("filename", help="Model file")
    presolve.add_argument(
        "-methods",
        nargs="+",
        default="PaPILO",
        choices=[m.name for m in PresolvingMethod],
        help="Technique to be used. If omitted, PaPILO is employed.",
    )
    app = subparsers.add_parser("app", help="Start the app ui.")

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    match args.command:
        case "app":
            start_ui()

        case "presolve":
            handle_presolve(args.filename, args.methods)


if __name__ == "__main__":
    main()
