"""esp ui — launch the PyQt6 constraint matrix analyser."""
from __future__ import annotations

import argparse
import sys


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser("ui", help="Launch the PyQt6 constraint matrix analyser.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from PyQt6.QtWidgets import QApplication
    from tools.constraint_matrix_analyser.app import ConstraintMatrixAnalyzer

    app = QApplication(sys.argv)
    window = ConstraintMatrixAnalyzer()
    window.show()
    sys.exit(app.exec())
