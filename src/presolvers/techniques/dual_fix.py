from __future__ import annotations

from typing import Dict

import mip
from mip import Var

from src.core.model import Model
from src.presolvers.algorithm import PresolveAlgorithm


class DualFix(PresolveAlgorithm):
    """
    Dual fixing presolver.

    For each variable x_k, count its up- and downlocks:
      - uplocks(k)   = number of constraints that forbid x_k from increasing
                       (i.e. a_k > 0 in a <= row, or a_k < 0 in a >= row)
      - downlocks(k) = number of constraints that forbid x_k from decreasing
                       (i.e. a_k < 0 in a <= row, or a_k > 0 in a >= row)

    Fixing rules (minimisation sense; signs flipped for maximisation):
      - uplocks(k) == 0 and c_k >= 0  =>  fix x_k at lb_k
            (moving x_k up cannot improve the objective and is unconstrained
             from above, so the optimum is at the lower bound)
      - downlocks(k) == 0 and c_k <= 0  =>  fix x_k at ub_k
            (moving x_k up can only improve the objective and is unconstrained
             from below, so the optimum is at the upper bound)

    Equality constraints contribute to *both* uplocks and downlocks, so
    variables that appear only in equalities are never fixed by this rule.
    """

    def __init__(self, model: Model):
        assert isinstance(model.model, mip.Model)
        super().__init__(model)
        self.name: str = "Dual Fix"
        self.mip: mip.Model = model.model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def presolve(self, max_rounds: int = 1, tol: float = 1e-9) -> int:
        """
        Apply dual fixing.

        Parameters
        ----------
        max_rounds
            Number of passes.  One pass is usually sufficient because fixing
            a variable does not create new dual-fix opportunities (unlike, say,
            bound tightening).  Additional rounds are supported for consistency
            with the rest of the framework.
        tol
            Numerical tolerance for coefficient and bound comparisons.

        Returns
        -------
        int
            Total number of variables fixed.
        """
        total_fixes = 0

        for _ in range(max_rounds):
            round_fixes = self._fix_pass(tol=tol)
            total_fixes += round_fixes

            if round_fixes == 0:
                break

        if total_fixes > 0:
            self.model.update_matrix()

        return total_fixes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fix_pass(self, tol: float) -> int:
        """One sweep over all variables; return the number fixed."""
        uplocks, downlocks = self._compute_locks(tol=tol)

        # Determine whether the problem is a minimisation or maximisation.
        # python-mip stores the sense on the model as mip.MINIMIZE / mip.MAXIMIZE.
        is_minimise = self.mip.sense == mip.MINIMIZE

        n_fixes = 0

        for var in self.mip.vars:
            if self._is_fixed(var, tol):
                continue

            c_k = float(var.obj)  # objective coefficient
            # Flip sign so we can always reason in terms of minimisation.
            if not is_minimise:
                c_k = -c_k

            up = uplocks.get(var.name, 0)
            dn = downlocks.get(var.name, 0)

            lb = float(var.lb)
            ub = float(var.ub)

            if up == 0 and c_k >= -tol:
                # Increasing x_k never violates any constraint and cannot
                # improve (minimise) the objective => fix at lower bound.
                if not _is_finite(lb):
                    # Unbounded below with zero uplocks: problem is unbounded.
                    # Leave it for the solver to detect.
                    continue
                var.lb = lb
                var.ub = lb
                n_fixes += 1

            elif dn == 0 and c_k <= tol:
                # Decreasing x_k never violates any constraint and cannot
                # worsen (minimise) the objective => fix at upper bound.
                if not _is_finite(ub):
                    continue
                var.lb = ub
                var.ub = ub
                n_fixes += 1

        return n_fixes

    def _compute_locks(self, tol: float) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Return (uplocks, downlocks) dicts keyed by variable name.

        For a normalised <= row:
          a_k > 0  =>  uplock   (x_k can't freely increase)
          a_k < 0  =>  downlock (x_k can't freely decrease)

        For a >= row the inequalities reverse.
        Equality rows contribute to both.
        """
        uplocks: Dict[str, int] = {}
        downlocks: Dict[str, int] = {}

        for constr in self.mip.constrs:
            expr = constr.expr
            sense = expr.sense  # "<" = <=,  ">" = >=,  "=" = ==

            for var, coef in expr.expr.items():
                a = float(coef)
                if abs(a) <= tol:
                    continue

                name = var.name

                if sense == "<":
                    # <= row: positive coef -> uplock, negative -> downlock
                    if a > 0:
                        uplocks[name] = uplocks.get(name, 0) + 1
                    else:
                        downlocks[name] = downlocks.get(name, 0) + 1

                elif sense == ">":
                    # >= row: positive coef -> downlock, negative -> uplock
                    if a > 0:
                        downlocks[name] = downlocks.get(name, 0) + 1
                    else:
                        uplocks[name] = uplocks.get(name, 0) + 1

                else:  # "=" — equality locks in both directions
                    uplocks[name] = uplocks.get(name, 0) + 1
                    downlocks[name] = downlocks.get(name, 0) + 1

        return uplocks, downlocks

    def _is_fixed(self, var: Var, tol: float = 1e-9) -> bool:
        """Return True if the variable is already fixed (lb == ub)."""
        return abs(float(var.lb) - float(var.ub)) <= tol


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------


def _is_finite(value: float) -> bool:
    import math
    import sys

    # python-mip stores mip.INF (math.inf) internally as sys.float_info.max,
    # so math.isfinite alone is insufficient — we also exclude that sentinel.
    return math.isfinite(value) and abs(value) < sys.float_info.max
