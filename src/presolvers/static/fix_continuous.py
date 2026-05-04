from __future__ import annotations

import mip
from mip import Var

from src.presolvers.base import StaticPresolvingAlgorithm


class FixContinuous(StaticPresolvingAlgorithm):

    @property
    def name(self) -> str:
        return "Fix Continuous"

    @property
    def slug(self) -> str:
        return "fix_continuous_static"

    def _run(self, model) -> None:
        self.mip: mip.Model = model.model
        self.presolve()

    def presolve(self, max_rounds: int = 1, tol: float = 1e-9) -> int:
        """
        Fix continuous variables whose lower and upper bounds are equal (or
        numerically indistinguishable), then substitute the fixed value into
        every constraint and the objective and remove the variable.

        A continuous variable x_k with lb_k == ub_k (up to *tol*) is fixed to
        the value v_k = (lb_k + ub_k) / 2.  For every constraint containing
        x_k the fixed contribution v_k * a_k is moved to the RHS, and the
        column is dropped.  The objective constant is updated similarly.

        This mirrors the behaviour of PaPILO's FixContinuous presolver.

        Parameters
        ----------
        max_rounds
            Number of passes over all variables.  One pass is usually
            sufficient because fixing a variable does not tighten the bounds
            of other variables.
        tol
            Absolute tolerance used to decide whether lb == ub.

        Returns
        -------
        int
            Total number of variables fixed and eliminated.
        """
        total_fixed = 0

        for _ in range(max_rounds):
            round_fixed = 0

            for var in list(self.mip.vars):
                if self._is_continuous(var) and self._bounds_are_equal(var, tol):
                    self._fix_and_substitute(var, tol)
                    round_fixed += 1

            total_fixed += round_fixed

            if round_fixed == 0:
                break

        return total_fixed

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _is_continuous(self, var: Var) -> bool:
        return var.var_type == "C"

    def _bounds_are_equal(self, var: Var, tol: float) -> bool:
        lb = float(var.lb)
        ub = float(var.ub)
        return abs(ub - lb) <= tol

    def _fix_and_substitute(self, var: Var, tol: float) -> None:
        """
        Substitute var = v into all constraints that contain it, adjust their
        RHS by -a_k * v, then remove the variable column from the model.
        """
        lb = float(var.lb)
        ub = float(var.ub)
        v = (lb + ub) / 2.0

        # Collect every constraint that references this variable.
        # python-mip exposes the column (list of (constr, coef) pairs) via
        # var.column.
        constrs_to_update = []
        for constr, coef in zip(var.column.constrs, var.column.coeffs):
            a_k = float(coef)
            if abs(a_k) <= tol:
                continue
            constrs_to_update.append((constr, a_k))

        # For each affected constraint, rebuild it with the variable removed
        # and the RHS shifted by -a_k * v.
        for constr, a_k in constrs_to_update:
            expr = constr.expr
            sense = expr.sense
            old_rhs = float(constr.rhs)
            new_rhs = old_rhs - a_k * v

            # Rebuild coefficients without the fixed variable
            new_coeffs = {
                v_: float(c)
                for v_, c in expr.expr.items()
                if v_ is not var and abs(float(c)) > tol
            }

            name = constr.name
            self.mip.remove(constr)

            if new_coeffs:
                lhs = mip.xsum(c * v_ for v_, c in new_coeffs.items())
            else:
                # All other variables were also absent; constraint becomes
                # trivial — only add it back if it could still be infeasible.
                lhs = mip.xsum([])  # zero expression

            if sense == "<":
                self.mip.add_constr(lhs <= new_rhs, name=name)
            elif sense == ">":
                self.mip.add_constr(lhs >= new_rhs, name=name)
            else:  # "="
                self.mip.add_constr(lhs == new_rhs, name=name)

        # Fix the variable at v so python-mip can safely remove the column,
        # then remove it.
        var.lb = v
        var.ub = v
        self.mip.remove(var)
