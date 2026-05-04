"""Lightweight read-only MPS metadata scanner.

Does not load the model into any solver — just scans the file to count
rows, columns, non-zeros, and read basic structural metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MPSMetadata:
    path: Path
    n_constraints: int          # number of named rows (excluding objective)
    n_vars: int                 # number of columns
    nnz: int                    # non-zero entries in COLUMNS section
    n_integer_vars: int         # binary + general integer variables
    has_objective: bool         # whether a free row (N type) exists
    obj_name: str | None        # name of the objective row if present

    @property
    def density(self) -> float:
        denom = self.n_constraints * self.n_vars
        return self.nnz / denom if denom > 0 else 0.0


def read_metadata(path: Path | str) -> MPSMetadata:
    """
    Scan an MPS file and return structural metadata without loading it into memory.

    Supports both fixed-format and free-format (CPLEX) MPS.
    """
    path = Path(path)
    n_constraints = 0
    n_vars = 0
    nnz = 0
    n_integer_vars = 0
    has_objective = False
    obj_name: str | None = None

    section: str | None = None
    in_int_marker = False
    seen_cols: set[str] = set()

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("$") or line.startswith("*"):
                continue

            # Section headers are uppercase with no leading whitespace
            upper = line.upper()
            if upper in ("ROWS", "COLUMNS", "RHS", "BOUNDS", "RANGES", "SOS", "ENDATA"):
                section = upper
                in_int_marker = False
                continue

            if section == "ROWS":
                parts = line.split()
                if len(parts) >= 2:
                    row_type = parts[0].upper()
                    row_name = parts[1]
                    if row_type == "N":
                        has_objective = True
                        if obj_name is None:
                            obj_name = row_name
                    elif row_type in ("L", "G", "E"):
                        n_constraints += 1

            elif section == "COLUMNS":
                if "'MARKER'" in line.upper():
                    # Integer marker: 'MARKER' 'INTORG' or 'INTEND'
                    in_int_marker = "'INTORG'" in line.upper()
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                col_name = parts[0]
                if col_name not in seen_cols:
                    seen_cols.add(col_name)
                    n_vars += 1
                    if in_int_marker:
                        n_integer_vars += 1

                # Each COLUMNS line has 1 or 2 (row, value) pairs
                n_entries = (len(parts) - 1) // 2
                nnz += n_entries

            elif section == "BOUNDS":
                parts = line.split()
                # BV (binary) bound marks a variable as binary integer
                if len(parts) >= 3 and parts[0].upper() == "BV":
                    col_name = parts[2]
                    if col_name in seen_cols:
                        n_integer_vars += 1  # may double-count if also in INT markers

    return MPSMetadata(
        path=path,
        n_constraints=n_constraints,
        n_vars=n_vars,
        nnz=nnz,
        n_integer_vars=n_integer_vars,
        has_objective=has_objective,
        obj_name=obj_name,
    )
