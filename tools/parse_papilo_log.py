"""Parse a PaPILO verbose log and print which presolvers fired in which order.

Usage:
    python tools/parse_papilo_log.py papilo.txt
    python tools/parse_papilo_log.py < papilo.txt
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field


_MODEL_RE = re.compile(r"starting presolve of problem\s+(\S+)")

_ROUND_RE = re.compile(
    r"^round\s+(\d+)\s+\(\s*(\w+)\s*\):\s*"
    r"(\d+) del cols,\s*(\d+) del rows,\s*(\d+) chg bounds,\s*"
    r"(\d+) chg sides,\s*(\d+) chg coeffs,\s*(\d+) tsx applied,\s*(\d+) tsx conflicts"
)
_PRESOLVER_RE = re.compile(
    r"^Presolver\s+(\w+)\s+applying"
)


@dataclass
class Deltas:
    del_cols: int
    del_rows: int
    chg_bounds: int
    chg_sides: int
    chg_coeffs: int
    tsx_applied: int
    tsx_conflicts: int

    def diff(self, prev: "Deltas") -> "Deltas":
        """Return the per-round increment relative to the previous round's cumulative totals."""
        return Deltas(
            del_cols      = self.del_cols      - prev.del_cols,
            del_rows      = self.del_rows      - prev.del_rows,
            chg_bounds    = self.chg_bounds    - prev.chg_bounds,
            chg_sides     = self.chg_sides     - prev.chg_sides,
            chg_coeffs    = self.chg_coeffs    - prev.chg_coeffs,
            tsx_applied   = self.tsx_applied   - prev.tsx_applied,
            tsx_conflicts = self.tsx_conflicts - prev.tsx_conflicts,
        )

    def is_empty(self) -> bool:
        return all(v == 0 for v in vars(self).values())


@dataclass
class Round:
    number: int
    kind: str
    cumulative: Deltas
    delta: Deltas        # change relative to previous round
    presolvers: list[str] = field(default_factory=list)


# Maps PaPILO internal presolver name → esp CLI method key.
_PAPILO_TO_CLI: dict[str, str] = {
    "coefftightening": "coeff_strengthening",
    "propagation":     "propagation",
    "colsingleton":    "col_singleton",
    "dualfix":         "dual_fix",
    "fixcontinuous":   "fix_continuous",
    "parallelcols":    "parallel_cols",
    "parallelrows":    "parallel_rows",
    "simpleprobing":   "simple_probing",
    "doubletoneq":     "double_to_neq",
    "simplifyineq":    "simplify_ineq",
    "stuffing":        "stuffing",
    "domcol":          "dom_col",
    "dualinfer":       "dual_infer",
    "implint":         "impl_int",
    "probing":         "probing",
    "sparsify":        "sparsify",
    "cliquemerging":   "clique_merging",
    "substitution":    "substitution",
}


def parse(lines: list[str]) -> tuple[list[Round], str | None]:
    rounds: list[Round] = []
    current: Round | None = None
    prev_deltas = Deltas(0, 0, 0, 0, 0, 0, 0)
    model_path: str | None = None

    for line in lines:
        if model_path is None:
            m = _MODEL_RE.search(line)
            if m:
                model_path = m.group(1)

        m = _ROUND_RE.match(line)
        if m:
            cumulative = Deltas(
                del_cols      = int(m.group(3)),
                del_rows      = int(m.group(4)),
                chg_bounds    = int(m.group(5)),
                chg_sides     = int(m.group(6)),
                chg_coeffs    = int(m.group(7)),
                tsx_applied   = int(m.group(8)),
                tsx_conflicts = int(m.group(9)),
            )
            delta = cumulative.diff(prev_deltas)
            current = Round(
                number     = int(m.group(1)),
                kind       = m.group(2),
                cumulative = cumulative,
                delta      = delta,
            )
            rounds.append(current)
            prev_deltas = cumulative
            continue

        m = _PRESOLVER_RE.match(line)
        if m and current is not None:
            current.presolvers.append(m.group(1))

    return rounds, model_path


def _fmt_delta(d: Deltas) -> str:
    parts = []
    if d.del_cols:      parts.append(f"del_cols={d.del_cols:+d}")
    if d.del_rows:      parts.append(f"del_rows={d.del_rows:+d}")
    if d.chg_bounds:    parts.append(f"chg_bounds={d.chg_bounds:+d}")
    if d.chg_sides:     parts.append(f"chg_sides={d.chg_sides:+d}")
    if d.chg_coeffs:    parts.append(f"chg_coeffs={d.chg_coeffs:+d}")
    if d.tsx_applied:   parts.append(f"tsx={d.tsx_applied:+d}")
    if d.tsx_conflicts: parts.append(f"conflicts={d.tsx_conflicts:+d}")
    return "  ".join(parts) if parts else "no changes"


def summarise(rounds: list[Round]) -> None:
    if not rounds:
        print("No rounds found.")
        return

    for r in rounds:
        label = f"Round {r.number} ({r.kind})"
        print(f"{label}:")
        print(f"  Δ {_fmt_delta(r.delta)}")
        if not r.presolvers:
            print("  presolvers: —")
        else:
            collapsed: list[str] = []
            for name in r.presolvers:
                if collapsed and collapsed[-1].split(" ×")[0] == name:
                    base, _, count = collapsed[-1].partition(" ×")
                    collapsed[-1] = f"{base} ×{int(count or 1) + 1}"
                else:
                    collapsed.append(name)
            print(f"  presolvers: {', '.join(collapsed)}")
        print()


def esp_command(rounds: list[Round], model_path: str | None) -> str:
    method_specs: list[str] = []
    for r in rounds:
        for name in r.presolvers:
            cli_key = _PAPILO_TO_CLI.get(name)
            if cli_key is None:
                cli_key = name  # fallback: use raw name and let esp error
            method_specs.append(f"{cli_key}:papilo")

    model_arg = model_path if model_path else "<model>"
    methods = " ".join(method_specs)
    return f"esp presolve --model {model_arg} --method {methods}"


def to_metadata(rounds: list[Round], model_path: str | None) -> dict:
    last = rounds[-1].cumulative if rounds else Deltas(0, 0, 0, 0, 0, 0, 0)
    return {
        "papilo": {
            "model_path": model_path,
            "n_rounds": len(rounds),
            "total": vars(last),
            "presolver_sequence": [p for r in rounds for p in r.presolvers],
            "rounds": [
                {
                    "number": r.number,
                    "kind": r.kind,
                    "presolvers": r.presolvers,
                    "delta": vars(r.delta),
                }
                for r in rounds
            ],
        }
    }


def save_metadata(meta: dict, output_dir: str) -> None:
    import json
    from pathlib import Path

    path = Path(output_dir) / "metadata.json"
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(meta)
    path.write_text(json.dumps(existing, indent=2))
    print(f"Metadata written to {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="PaPILO log file (default: stdin).")
    parser.add_argument("--output", "-o", default=None, metavar="DIR",
                        help="Directory to write/merge metadata.json into.")
    args = parser.parse_args()

    if args.log:
        with open(args.log) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    rounds, model_path = parse(lines)
    summarise(rounds)

    print("Flat order (round, presolver, per-round deltas):")
    print(f"  {'round':<6} {'kind':<12} {'presolver':<20} {'del_cols':>9} {'del_rows':>9} {'chg_bounds':>11} {'tsx':>9}")
    print("  " + "-" * 72)
    for r in rounds:
        d = r.delta
        for name in r.presolvers:
            print(f"  {r.number:<6} {r.kind:<12} {name:<20} "
                  f"{d.del_cols:>9} {d.del_rows:>9} {d.chg_bounds:>11} {d.tsx_applied:>9}")

    print()
    print("esp command:")
    print(f"  {esp_command(rounds, model_path)}")

    if args.output:
        save_metadata(to_metadata(rounds, model_path), args.output)


if __name__ == "__main__":
    main()
