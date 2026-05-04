from __future__ import annotations

import math
from typing import Dict, Tuple

import mip
from mip import Constr, Var

from src.presolvers.base import StaticPresolvingAlgorithm


class ConstraintPropagation(StaticPresolvingAlgorithm):

    @property
    def name(self) -> str:
        return "Constraint Propagation"

    @property
    def slug(self) -> str:
        return "propagation_static"

    def _run(self, model) -> None:
        self.mip: mip.Model = model.model
        self.presolve()

    def presolve(self, max_rounds: int = 10, tol: float = 1e-9) -> int:
        """
        Apply constraint propagation (bounds tightening).

        For each constraint and each variable in that constraint, the activity
        of the remaining terms is used to derive tighter bounds on that
        variable.  The process is repeated until no further tightening occurs
        or ``max_rounds`` is exhausted.

        Returns
        -------
        int
            Total number of bound tightenings applied.
        """
        total_changes = 0

        for _ in range(max_rounds):
            round_changes = 0

            for constr in self.mip.constrs:
                round_changes += self._propagate_constraint(constr, tol=tol)

            total_changes += round_changes

            if round_changes == 0:
                break

        return total_changes

    def _propagate_constraint(self, constr: Constr, tol: float = 1e-9) -> int:
        expr = constr.expr
        sense = expr.sense

        coeffs: Dict[Var, float] = {
            var: float(coef)
            for var, coef in expr.expr.items()
            if abs(float(coef)) > tol
        }

        if len(coeffs) <= 1:
            return 0

        rhs = float(constr.rhs)
        n_changes = 0
        senses_to_process = ["<", ">"] if sense == "=" else [sense]

        for normalised_sense in senses_to_process:
            if normalised_sense == ">":
                work_coeffs = {var: -coef for var, coef in coeffs.items()}
                work_rhs = -rhs
            else:
                work_coeffs = dict(coeffs)
                work_rhs = rhs

            for var in work_coeffs:
                n_changes += self._tighten_var_bounds(var, work_coeffs, work_rhs, tol=tol)

        return n_changes

    def _tighten_var_bounds(
        self,
        var: Var,
        coeffs: Dict[Var, float],
        rhs: float,
        tol: float = 1e-9,
    ) -> int:
        a_k = coeffs[var]
        if abs(a_k) <= tol:
            return 0

        min_act, ok = self._min_activity_excluding(coeffs, skip_var=var, tol=tol)
        if not ok:
            return 0

        implied = (rhs - min_act) / a_k

        if a_k > 0:
            new_ub = implied
            if self._is_integral_var(var):
                new_ub = math.floor(new_ub + tol)
            current_ub = float(var.ub)
            if not math.isfinite(current_ub) or new_ub < current_ub - tol:
                if new_ub >= float(var.lb) - tol:
                    var.ub = new_ub
                    return 1
        else:
            new_lb = implied
            if self._is_integral_var(var):
                new_lb = math.ceil(new_lb - tol)
            current_lb = float(var.lb)
            if not math.isfinite(current_lb) or new_lb > current_lb + tol:
                if new_lb <= float(var.ub) + tol:
                    var.lb = new_lb
                    return 1

        return 0

    def _min_activity_excluding(
        self,
        coeffs: Dict[Var, float],
        skip_var: Var,
        tol: float = 1e-9,
    ) -> Tuple[float, bool]:
        total = 0.0

        for var, coef in coeffs.items():
            if var is skip_var or abs(coef) <= tol:
                continue

            if coef > 0.0:
                bound = float(var.lb)
                if not math.isfinite(bound):
                    return 0.0, False
                total += coef * bound
            else:
                bound = float(var.ub)
                if not math.isfinite(bound):
                    return 0.0, False
                total += coef * bound

        return total, True

    def _is_integral_var(self, var: Var) -> bool:
        return var.var_type in ("B", "I")
