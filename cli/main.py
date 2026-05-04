"""esp — energy-system-preprocessing CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="esp",
        description="Energy-system preprocessing toolchain.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register subcommands
    from cli.generate import add_subcommand as add_generate
    from cli.presolve import add_subcommand as add_presolve
    from cli.detect import add_subcommand as add_detect
    from cli.compare import add_subcommand as add_compare
    from cli.solve import add_subcommand as add_solve
    from cli.ui import add_subcommand as add_ui

    add_generate(subparsers)
    add_presolve(subparsers)
    add_detect(subparsers)
    add_compare(subparsers)
    add_solve(subparsers)
    add_ui(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
