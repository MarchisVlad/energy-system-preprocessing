from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import mip
from mip import Constr, Var

from src.presolvers.base import StaticPresolvingAlgorithm


class CoefficientStrengthening(StaticPresolvingAlgorithm):

    @property
    def name(self) -> str:
        return "Coefficient Strengthening"

    @property
    def slug(self) -> str:
        return "coeff_strengthening_static"

    def _run(self, model) -> None:
        self.mip: mip.Model = model.model
        self.presolve()

    def presolve(self, max_rounds: int = 1, tol: float = 1e-9):
        """
        Apply Achterberg-style coefficient tightening.

        This implements the single-row, single-variable rule on each one-sided row:
            a^T x <= b   or   a^T x >= b

        For each integral variable x_k in a row, using the row-wise bound
        u_iS on the remaining variables:
            if a_k > 0:
                d = b - u_iS - a_k * (ub_k - 1)
                if 0 < d <= a_k:
                    a_k <- a_k - d
                    b   <- b - d * ub_k

            if a_k < 0:
                d = b - u_iS - a_k * (lb_k + 1)
                if 0 < d <= -a_k:
                    a_k <- a_k + d
                    b   <- b + d * lb_k

        Returns
        -------
        int
            Total number of coefficient tightenings performed.
        """

        total_changes = 0

        for _ in range(max_rounds):
            round_changes = 0
            changes: list[tuple[int, str, Dict[Var, float], float, str]] = []

            # Analyze all constraints first, then apply any modifications in a
            # separate pass. This avoids mutating the model while iterating
            # over its live constraint list, which can cause row-index issues
            # in the underlying CBC backend.
            constraints = list(self.mip.constrs)
            for i, constr in enumerate(constraints):
                result = self._analyse_constraint(constr, tol=tol)
                if result is not None:
                    _, name, coeffs, rhs, sense = result
                    changes.append((i, name, coeffs, rhs, sense))

            if changes:
                # Remove modified rows in reverse order to preserve valid row
                # indices while deleting from the original model.
                for i, name, coeffs, rhs, sense in reversed(changes):
                    constr = constraints[i]
                    self.mip.remove(constr)

                # Re-add modified rows after all deletions are complete.
                for _, name, coeffs, rhs, sense in changes:
                    lhs = mip.xsum(coef * var for var, coef in coeffs.items())
                    if sense == "<":
                        self.mip.add_constr(lhs <= rhs, name=name)
                    else:
                        self.mip.add_constr(lhs >= rhs, name=name)

                round_changes = len(changes)

            total_changes += round_changes

            if round_changes == 0:
                break

        return total_changes

    def _analyse_constraint(
        self, constr: Constr, tol: float = 1e-9
    ) -> Optional[Tuple[Constr, str, Dict[Var, float], float, str]]:
        expr = constr.expr
        sense = expr.sense

        if sense not in ("<", ">"):
            return None  # skip equality rows

        coeffs: Dict[Var, float] = {
            var: float(coef)
            for var, coef in expr.expr.items()
            if abs(float(coef)) > tol
        }

        if len(coeffs) <= 1:
            return None

        rhs = float(constr.rhs)

        normalized_from_ge = sense == ">"
        if normalized_from_ge:
            coeffs = {var: -coef for var, coef in coeffs.items()}
            rhs = -rhs

        n_changes = 0

        for var in list(coeffs.keys()):
            if not self._is_integral_var(var):
                continue

            if self._is_fixed(var, tol):
                continue

            a_k = coeffs[var]
            if abs(a_k) <= tol:
                continue

            u_iS, ok = self._max_activity_excluding(coeffs, skip_var=var, tol=tol)
            if not ok:
                continue

            if a_k > 0:
                u_k = float(var.ub)
                if not math.isfinite(u_k):
                    continue

                d = rhs - u_iS - a_k * (u_k - 1.0)

                if d <= tol or d > a_k + tol:
                    continue

                coeffs[var] = a_k - d
                rhs = rhs - d * u_k
                n_changes += 1

            else:
                l_k = float(var.lb)
                if not math.isfinite(l_k):
                    continue

                d = rhs - u_iS - a_k * (l_k + 1.0)

                if d <= tol or d > (-a_k) + tol:
                    continue

                coeffs[var] = a_k + d
                rhs = rhs + d * l_k
                n_changes += 1

            if abs(coeffs[var]) <= tol:
                coeffs[var] = 0.0

        if n_changes == 0:
            return None

        coeffs = {var: coef for var, coef in coeffs.items() if abs(coef) > tol}

        if normalized_from_ge:
            coeffs = {var: -coef for var, coef in coeffs.items()}
            rhs = -rhs

        return constr, constr.name, coeffs, rhs, sense

    def _max_activity_excluding(
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
                bound = float(var.ub)
                if not math.isfinite(bound):
                    return 0.0, False
                total += coef * bound
            else:
                bound = float(var.lb)
                if not math.isfinite(bound):
                    return 0.0, False
                total += coef * bound

        return total, True

    def _is_integral_var(self, var: Var) -> bool:
        return var.var_type in ("B", "I")

    def _is_fixed(self, var: Var, tol: float = 1e-9) -> bool:
        return abs(float(var.lb) - float(var.ub)) <= tol
