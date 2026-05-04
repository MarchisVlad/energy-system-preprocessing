from __future__ import annotations

import math
import sys
from typing import Dict

import mip
from mip import Var

from src.presolvers.base import StaticPresolvingAlgorithm


class DualFix(StaticPresolvingAlgorithm):
    """
    Dual fixing presolver.

    For each variable x_k, count its up- and downlocks:
      - uplocks(k)   = number of constraints that forbid x_k from increasing
      - downlocks(k) = number of constraints that forbid x_k from decreasing

    Fixing rules (minimisation; signs flipped for maximisation):
      - uplocks(k) == 0 and c_k >= 0  =>  fix x_k at lb_k
      - downlocks(k) == 0 and c_k <= 0  =>  fix x_k at ub_k
    """

    @property
    def name(self) -> str:
        return "Dual Fix"

    @property
    def slug(self) -> str:
        return "dual_fix_static"

    def _run(self, model) -> None:
        self.mip: mip.Model = model.model
        self.presolve()

    def presolve(self, max_rounds: int = 1, tol: float = 1e-9) -> int:
        """
        Apply dual fixing.

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

        return total_fixes

    def _fix_pass(self, tol: float) -> int:
        uplocks, downlocks = self._compute_locks(tol=tol)
        is_minimise = self.mip.sense == mip.MINIMIZE
        n_fixes = 0

        for var in self.mip.vars:
            if self._is_fixed(var, tol):
                continue

            c_k = float(var.obj)
            if not is_minimise:
                c_k = -c_k

            up = uplocks.get(var.name, 0)
            dn = downlocks.get(var.name, 0)
            lb = float(var.lb)
            ub = float(var.ub)

            if up == 0 and c_k >= -tol:
                if not _is_finite(lb):
                    continue
                var.lb = lb
                var.ub = lb
                n_fixes += 1

            elif dn == 0 and c_k <= tol:
                if not _is_finite(ub):
                    continue
                var.lb = ub
                var.ub = ub
                n_fixes += 1

        return n_fixes

    def _compute_locks(self, tol: float) -> tuple[Dict[str, int], Dict[str, int]]:
        uplocks: Dict[str, int] = {}
        downlocks: Dict[str, int] = {}

        for constr in self.mip.constrs:
            expr = constr.expr
            sense = expr.sense

            for var, coef in expr.expr.items():
                a = float(coef)
                if abs(a) <= tol:
                    continue

                name = var.name

                if sense == "<":
                    if a > 0:
                        uplocks[name] = uplocks.get(name, 0) + 1
                    else:
                        downlocks[name] = downlocks.get(name, 0) + 1

                elif sense == ">":
                    if a > 0:
                        downlocks[name] = downlocks.get(name, 0) + 1
                    else:
                        uplocks[name] = uplocks.get(name, 0) + 1

                else:
                    uplocks[name] = uplocks.get(name, 0) + 1
                    downlocks[name] = downlocks.get(name, 0) + 1

        return uplocks, downlocks

    def _is_fixed(self, var: Var, tol: float = 1e-9) -> bool:
        return abs(float(var.lb) - float(var.ub)) <= tol


def _is_finite(value: float) -> bool:
    # python-mip stores mip.INF as sys.float_info.max, so math.isfinite alone is insufficient
    return math.isfinite(value) and abs(value) < sys.float_info.max
