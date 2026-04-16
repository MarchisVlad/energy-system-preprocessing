from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import mip
from mip import Constr, Var

from src.core.model import Model
from src.presolvers.algorithm import PresolveAlgorithm


class ConstraintPropagation(PresolveAlgorithm):

    def __init__(self, model: Model):
        assert isinstance(model.model, mip.Model)
        super().__init__(model)
        self.name: str = "Constraint Propagation"
        self.mip: mip.Model = model.model

    def presolve(self, max_rounds: int = 10, tol: float = 1e-9) -> int:
        """
        Apply constraint propagation (bounds tightening).

        For each constraint and each variable in that constraint, the activity
        of the remaining terms is used to derive tighter bounds on that
        variable.  The process is repeated until no further tightening occurs
        or ``max_rounds`` is exhausted.

        For a normalised row  a^T x <= b  and variable x_k:

            if a_k > 0:
                ub_k  <=  (b - min_activity_S) / a_k
            if a_k < 0:
                lb_k  >=  (b - min_activity_S) / a_k

        where  min_activity_S = sum_{j != k} a_j * lb_j  (a_j > 0)
                               + sum_{j != k} a_j * ub_j  (a_j < 0).

        Equality rows are treated as a "<=" row and a ">=" row simultaneously,
        so both upper and lower bounds can be tightened.

        Parameters
        ----------
        max_rounds
            Maximum number of passes over all constraints.
        tol
            Numerical tolerance used when deciding whether a bound has
            actually improved.

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

        if total_changes > 0:
            self.model.update_matrix()

        return total_changes

    def _propagate_constraint(self, constr: Constr, tol: float = 1e-9) -> int:
        """
        Tighten variable bounds implied by a single constraint.

        Returns the number of bound improvements made.
        """
        expr = constr.expr
        sense = expr.sense  # "<" for <=, ">" for >=, "=" for ==

        # Extract nonzero coefficients
        coeffs: Dict[Var, float] = {
            var: float(coef)
            for var, coef in expr.expr.items()
            if abs(float(coef)) > tol
        }

        if len(coeffs) <= 1:
            return 0

        rhs = float(constr.rhs)

        n_changes = 0

        # Equality rows can tighten bounds in both directions.
        senses_to_process = ["<", ">"] if sense == "=" else [sense]

        for normalised_sense in senses_to_process:
            # Normalise to <= by negating when sense is >=
            if normalised_sense == ">":
                work_coeffs = {var: -coef for var, coef in coeffs.items()}
                work_rhs = -rhs
            else:
                work_coeffs = dict(coeffs)
                work_rhs = rhs

            # Propagate each variable in the normalised <= row
            for var in work_coeffs:
                improvement = self._tighten_var_bounds(
                    var, work_coeffs, work_rhs, tol=tol
                )
                n_changes += improvement

        return n_changes

    def _tighten_var_bounds(
        self,
        var: Var,
        coeffs: Dict[Var, float],
        rhs: float,
        tol: float = 1e-9,
    ) -> int:
        """
        For the normalised row  a^T x <= rhs, derive an implied bound on
        ``var`` and update it if it is tighter than the current bound.

        Returns 1 if a bound was tightened, 0 otherwise.
        """
        a_k = coeffs[var]
        if abs(a_k) <= tol:
            return 0

        min_act, ok = self._min_activity_excluding(coeffs, skip_var=var, tol=tol)
        if not ok:
            return 0

        # Implied bound: a_k * x_k <= rhs - min_act
        implied = (rhs - min_act) / a_k

        if a_k > 0:
            # Upper-bound tightening
            new_ub = implied
            if self._is_integral_var(var):
                new_ub = math.floor(new_ub + tol)
            current_ub = float(var.ub)
            if not math.isfinite(current_ub) or new_ub < current_ub - tol:
                # Guard against infeasibility: don't push ub below lb
                if new_ub >= float(var.lb) - tol:
                    var.ub = new_ub
                    return 1

        else:
            # Lower-bound tightening  (a_k < 0 flips the inequality)
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
        """
        Compute the minimum activity of the row, excluding ``skip_var``:

            sum_{j != k} a_j * lb_j   if a_j > 0
            sum_{j != k} a_j * ub_j   if a_j < 0

        Returns
        -------
        (value, is_finite)
            ``is_finite`` is False when any required bound is infinite,
            meaning no useful implied bound can be derived.
        """
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
        """Treat binary and integer variables as integral."""
        return var.var_type in ("B", "I")
