from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from src.core.presolving import PresolveStatus, PresolvingMethod
from src.presolvers.base import PresolveResult, PresolvingAlgorithm

_BINARY = Path(__file__).parent / "papilo" / "build" / "bin" / "papilo"

# All presolver names as registered internally by PaPILO (from PresolveMethod::setName calls).
_ALL_PAPILO_PRESOLVERS = {
    "cliquemerging", "coefftightening", "colsingleton", "domcol", "doubletoneq",
    "dualfix", "dualinfer", "fixcontinuous", "implint", "parallelcols",
    "parallelrows", "probing", "propagation", "simpleprobing", "simplifyineq",
    "sparsify", "stuffing", "substitution",
}

# Maps PresolvingMethod enum → PaPILO internal presolver name.
_METHOD_TO_PAPILO_NAME: dict[PresolvingMethod, str] = {
    PresolvingMethod.CoeffTightening: "coefftightening",
    PresolvingMethod.Propagation:     "propagation",
    PresolvingMethod.ColSingleton:    "colsingleton",
    PresolvingMethod.DualFix:         "dualfix",
    PresolvingMethod.FixContinuous:   "fixcontinuous",
    PresolvingMethod.ParallelCols:    "parallelcols",
    PresolvingMethod.ParallelRows:    "parallelrows",
    PresolvingMethod.SimpleProbing:   "simpleprobing",
    PresolvingMethod.DoubleToNEq:     "doubletoneq",
    PresolvingMethod.SimpifyIneq:     "simplifyineq",
    PresolvingMethod.Stuffing:        "stuffing",
    PresolvingMethod.DomCol:          "domcol",
    PresolvingMethod.DualInfer:       "dualinfer",
    PresolvingMethod.ImplInt:         "implint",
    PresolvingMethod.Probing:         "probing",
    PresolvingMethod.Sparsify:        "sparsify",
    PresolvingMethod.CliqueMerging:   "cliquemerging",
    PresolvingMethod.Substitution:    "substitution",
}


class PaPILO(PresolvingAlgorithm):
    """
    PaPILO C++ presolving backend.

    Parameters
    ----------
    methods
        List of PresolvingMethod values to enable. None means PaPILO runs
        with all its default methods enabled.
    """

    def __init__(self, methods: list[PresolvingMethod] | None = None):
        self._methods = methods

    @property
    def name(self) -> str:
        if self._methods:
            return f"PaPILO({', '.join(m.name for m in self._methods)})"
        return "PaPILO"

    @property
    def slug(self) -> str:
        if self._methods:
            suffix = "_".join(m.name.lower() for m in self._methods)
            return f"{suffix}_papilo"
        return "papilo"

    def apply(self, mps_in: Path, output_path: Path) -> PresolveResult:
        if not _BINARY.exists():
            raise RuntimeError(
                f"PaPILO binary not found at {_BINARY}.\n"
                "Build it with:\n"
                "  cd src/presolvers/papilo && mkdir -p build && "
                "cd build && cmake .. && make")

        # Count before running — mps_in may equal output_path if caller reuses the path.
        n_before = _count_mps_rows_cols(mps_in)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(_BINARY), "presolve", "-f", str(mps_in), "-r", str(output_path)]

        settings_file = None
        if self._methods:
            # Disable every presolver except the ones requested, and cap to 1 round.
            enabled_names = {_METHOD_TO_PAPILO_NAME[m] for m in self._methods}
            disabled = _ALL_PAPILO_PRESOLVERS - enabled_names
            settings_lines = [f"{name}.enabled = false" for name in sorted(disabled)]
            settings_lines.append("presolve.maxrounds = 1")
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".set", delete=False)
            tf.write("\n".join(settings_lines) + "\n")
            tf.close()
            settings_file = tf.name
            cmd += ["-p", settings_file]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            if settings_file:
                Path(settings_file).unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"PaPILO failed for '{self.slug}':\n{result.stderr}")

        if not output_path.exists():
            raise RuntimeError(
                f"PaPILO produced no output for '{self.slug}'.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

        n_after = _count_mps_rows_cols(output_path)

        changed = n_after != n_before
        status = PresolveStatus.kReduced if changed else PresolveStatus.kUnchanged

        return PresolveResult(
            output_mps=output_path,
            status=status,
            n_vars_before=n_before[1],
            n_vars_after=n_after[1],
            n_constraints_before=n_before[0],
            n_constraints_after=n_after[0],
            nnz_before=-1,
            nnz_after=-1,
            debug_log=result.stdout,
        )


def _count_mps_rows_cols(path: Path) -> tuple[int, int]:
    """Quick scan of MPS sections to count rows and columns."""
    rows, cols = 0, 0
    section = None
    try:
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("$"):
                    continue
                if stripped in ("ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"):
                    section = stripped
                    continue
                if section == "ROWS" and stripped[0] in ("L", "G", "E", "N"):
                    rows += 1
                elif section == "COLUMNS":
                    cols += 1
    except OSError:
        pass
    return rows, cols
