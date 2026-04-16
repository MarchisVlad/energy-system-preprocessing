from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import mip
from mip import Constr, Var

from src.core.model import Model
from src.presolvers.algorithm import PresolveAlgorithm


class CoefficientStrengthening(PresolveAlgorithm):

    def __init__(self, model: Model):
        assert isinstance(model.model, mip.Model)
        super().__init__(model)
        self.name: str = "Coefficient Strengthening"
        self.mip: mip.Model = model.model

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

        Parameters
        ----------
        model
            A Model.
        max_rounds
            Number of presolve passes over all constraints.
        tol
            Numerical tolerance.

        Returns
        -------
        int
            Total number of coefficient tightenings performed.
        """

        total_changes = 0

        for _ in range(max_rounds):
            round_changes = 0

            # Iterate from the last constraint to the first, accessing each
            # by its *live* index rather than a pre-captured Constr object.
            # Removing at index i only shifts rows i+1..n-1, which we have
            # already visited; rows 0..i-1 are unaffected and are accessed
            # correctly on the next decrement.
            i = len(self.mip.constrs) - 1
            while i >= 0:
                constr = self.mip.constrs[i]
                result = self._analyse_constraint(constr, tol=tol)
                if result is not None:
                    _, name, coeffs, rhs, sense = result
                    self.mip.remove(constr)
                    lhs = mip.xsum(coef * var for var, coef in coeffs.items())
                    if sense == "<":
                        self.mip.add_constr(lhs <= rhs, name=name)
                    else:
                        self.mip.add_constr(lhs >= rhs, name=name)
                    round_changes += 1
                i -= 1

            total_changes += round_changes

            if round_changes == 0:
                break

        if total_changes > 0:
            self.model.update_matrix()

        return total_changes

    def _analyse_constraint(
        self, constr: Constr, tol: float = 1e-9
    ) -> Optional[Tuple[Constr, str, Dict[Var, float], float, str]]:
        """
        Compute tightened coefficients for *constr* without modifying the model.

        Returns ``(constr, new_coeffs, new_rhs, sense)`` if any coefficient was
        tightened, or ``None`` if the constraint needs no change.
        """
        expr = constr.expr
        sense = expr.sense

        # python-mip uses "<" for <=, ">" for >=, "=" for equality
        if sense not in ("<", ">"):
            return None  # skip equality rows

        # Extract nonzero coefficients
        coeffs: Dict[Var, float] = {
            var: float(coef)
            for var, coef in expr.expr.items()
            if abs(float(coef)) > tol
        }

        if len(coeffs) <= 1:
            return None

        rhs = float(constr.rhs)

        # Normalize >= rows to <= rows by multiplying by -1
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

            else:  # a_k < 0
                l_k = float(var.lb)
                if not math.isfinite(l_k):
                    continue

                d = rhs - u_iS - a_k * (l_k + 1.0)

                if d <= tol or d > (-a_k) + tol:
                    continue

                coeffs[var] = a_k + d
                rhs = rhs + d * l_k
                n_changes += 1

            # Clean tiny numerical residue
            if abs(coeffs[var]) <= tol:
                coeffs[var] = 0.0

        if n_changes == 0:
            return None

        # Remove numerically zero coefficients
        coeffs = {var: coef for var, coef in coeffs.items() if abs(coef) > tol}

        # Undo normalization if the original row was >=
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
        """
        Compute u_iS = max activity of the row excluding skip_var:
            sum_{j != k} a_j * ub_j   if a_j > 0
            sum_{j != k} a_j * lb_j   if a_j < 0

        Returns
        -------
        (value, is_finite)
        """
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
        """Treat binary and integer variables as integral."""
        return var.var_type in ("B", "I")

    def _is_fixed(self, var: Var, tol: float = 1e-9) -> bool:
        """Check whether lb == ub up to tolerance."""
        return abs(float(var.lb) - float(var.ub)) <= tol
