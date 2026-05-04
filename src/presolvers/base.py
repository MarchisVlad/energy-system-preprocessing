from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from src.core.presolving import PresolveStatus


@dataclass
class PresolveResult:
    output_mps: Path
    status: PresolveStatus
    n_vars_before: int
    n_vars_after: int
    n_constraints_before: int
    n_constraints_after: int
    nnz_before: int
    nnz_after: int
    debug_log: str = field(default="")

    @property
    def vars_reduced(self) -> int:
        return self.n_vars_before - self.n_vars_after

    @property
    def constraints_reduced(self) -> int:
        return self.n_constraints_before - self.n_constraints_after

    @property
    def nnz_reduced(self) -> int:
        return self.nnz_before - self.nnz_after


class PresolvingAlgorithm(ABC):
    """Base class for all presolving algorithms (static Python or external tools)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Coefficient Strengthening'."""
        ...

    @property
    @abstractmethod
    def slug(self) -> str:
        """Machine-readable key used in filenames, e.g. 'coeff_strengthening_static'."""
        ...

    @abstractmethod
    def apply(self, mps_in: Path, output_path: Path) -> PresolveResult:
        """Apply the technique to *mps_in* and write the result to *output_path*."""
        ...

    def postsolve_info(self) -> dict:
        return {}


class StaticPresolvingAlgorithm(PresolvingAlgorithm):
    """Mixin for Python-implemented presolvers that work on a loaded mip.Model."""

    def apply(self, mps_in: Path, output_path: Path) -> PresolveResult:
        import mip as _mip

        from src.core.model import Model

        model = Model(path=str(mps_in))
        assert isinstance(model.model, _mip.Model)

        n_vars_before = len(model.model.vars)
        n_constrs_before = len(model.model.constrs)
        nnz_before = int(model.A.nnz)

        self._run(model)
        model.update_matrix()

        n_vars_after = len(model.model.vars)
        n_constrs_after = len(model.model.constrs)
        nnz_after = int(model.A.nnz)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.model.write(str(output_path))

        changed = (n_vars_after, n_constrs_after, nnz_after) != (
            n_vars_before,
            n_constrs_before,
            nnz_before,
        )
        status = PresolveStatus.kReduced if changed else PresolveStatus.kUnchanged

        return PresolveResult(
            output_mps=output_path,
            status=status,
            n_vars_before=n_vars_before,
            n_vars_after=n_vars_after,
            n_constraints_before=n_constrs_before,
            n_constraints_after=n_constrs_after,
            nnz_before=nnz_before,
            nnz_after=nnz_after,
        )

    @abstractmethod
    def _run(self, model) -> None:
        """Apply the technique to *model* in-place."""
        ...
