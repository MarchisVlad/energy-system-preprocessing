from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mip
from mip import Constr, LinExpr, Var

from src.core.model import Model
from src.presolvers.algorithm import PresolveAlgorithm


class Sparsify(PresolveAlgorithm):
    """
    Non-zero cancellation ("sparsify") presolver.

    For an equality row q and another row r, find a scalar s such that
    replacing row r by (row_r - s * row_q) reduces the total number of
    non-zeros in the constraint matrix.

    Partition of supp(A_q) ∪ supp(A_r):
        S = { j in supp(A_q) ∩ supp(A_r) : s * A_qj == A_rj }  (cancels in r)
        T = { j in supp(A_q) ∩ supp(A_r) : s * A_qj != A_rj }  (modified in r)
        U = { j in supp(A_q) \\ supp(A_r) }                     (introduced in r)
        V = { j in supp(A_r) \\ supp(A_q) }                     (untouched in r)

    Net NNZ change in r is |U| - |S|, so apply when |S| - |U| >= min_gain.
    Only an equality row may be added to another row; otherwise the feasible
    region would change.

    Reference: Achterberg et al., "Presolve Reductions in Mixed Integer
    Programming" (2019); PaPILO's Sparsify presolver.
    """

    def __init__(self, model: Model):
        assert isinstance(model.model, mip.Model)
        super().__init__(model)
        self.name: str = "Sparsify"
        self.mip: mip.Model = model.model

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def presolve(
        self,
        max_rounds: int = 1,
        tol: float = 1e-9,
        ratio_tol: float = 1e-9,
        min_pivot: float = 1e-7,
        max_s: float = 1e6,
        min_gain: int = 1,
        **kwargs,
    ) -> Model:
        """
        Apply sparsify reductions.

        Parameters
        ----------
        max_rounds
            Number of presolve passes. Each pass modifies each row at most
            once to avoid stacking conflicting reductions.
        tol
            Numerical tolerance for treating coefficients as zero.
        ratio_tol
            Tolerance for bucketing ratios A_rj / A_qj when selecting s.
        min_pivot
            Reject any pivot coefficient A_qj with |A_qj| < min_pivot.
        max_s
            Reject any scalar s with |s| > max_s (prevents coefficient blow-up).
        min_gain
            Minimum NNZ reduction required per row modification.

        Returns
        -------
        Model
            The presolved model (modified in place).
        """
        total_changes = 0

        for _ in range(max_rounds):
            round_changes = self._run_pass(
                tol=tol,
                ratio_tol=ratio_tol,
                min_pivot=min_pivot,
                max_s=max_s,
                min_gain=min_gain,
            )
            total_changes += round_changes
            if round_changes == 0:
                break

        if total_changes > 0:
            self.model.update_matrix()

        return self.model

    # ------------------------------------------------------------------ #
    # One pass: collect all reductions, then apply
    # ------------------------------------------------------------------ #
    def _run_pass(
        self,
        tol: float,
        ratio_tol: float,
        min_pivot: float,
        max_s: float,
        min_gain: int,
    ) -> int:
        row_support, col_support = self._build_supports(tol=tol)

        # Snapshot constraints so we iterate over a stable list and can
        # reference rows by index during the remove/re-add phase.
        constraints: List[Constr] = list(self.mip.constrs)
        row_index: Dict[Constr, int] = {c: i for i, c in enumerate(constraints)}

        # Equation rows sorted by ascending support — smaller supports give
        # fewer ratio buckets and cheaper inner loops.
        eq_rows: List[Constr] = sorted(
            (c for c in constraints if self._is_equality(c)),
            key=lambda c: len(row_support[c]),
        )

        modified_rows: set = set()
        # Each entry: (row_index_of_r, new_name, new_coeffs, new_rhs, sense)
        changes: List[Tuple[int, str, Dict[Var, float], float, str]] = []

        for q in eq_rows:
            q_coeffs = row_support[q]
            if len(q_coeffs) < 2:
                continue

            # Candidate partner rows share at least one variable with q.
            # Pivoting on the variable with smallest column support keeps
            # the candidate set tight.
            pivot_var = min(q_coeffs, key=lambda v: len(col_support[v]))
            candidates = {
                r for r in col_support[pivot_var]
                if r is not q and r not in modified_rows
            }

            for r in candidates:
                if r in modified_rows:
                    continue
                result = self._best_combination(
                    q,
                    r,
                    q_coeffs,
                    row_support[r],
                    tol=tol,
                    ratio_tol=ratio_tol,
                    min_pivot=min_pivot,
                    max_s=max_s,
                    min_gain=min_gain,
                )
                if result is None:
                    continue
                _s, new_coeffs, new_rhs, _gain = result
                changes.append(
                    (row_index[r], r.name, new_coeffs, new_rhs, r.expr.sense))
                modified_rows.add(r)

        if not changes:
            return 0

        # Apply: remove modified rows in reverse row-index order so the
        # earlier indices remain valid, then re-add the new rows.
        # This mirrors the approach used in CoefficientStrengthening.
        changes_by_index = sorted(changes, key=lambda t: t[0])
        for i, _name, _coeffs, _rhs, _sense in reversed(changes_by_index):
            self.mip.remove(constraints[i])

        for _i, name, coeffs, rhs, sense in changes_by_index:
            lhs = mip.xsum(coef * var for var, coef in coeffs.items())
            if sense == "=":
                self.mip.add_constr(lhs == rhs, name=name)
            elif sense == "<":
                self.mip.add_constr(lhs <= rhs, name=name)
            elif sense == ">":
                self.mip.add_constr(lhs >= rhs, name=name)
            else:
                raise ValueError(f"Unknown constraint sense: {sense!r}")

        return len(changes)

    # ------------------------------------------------------------------ #
    # Core: pick the best s for a (q, r) pair and build the new row
    # ------------------------------------------------------------------ #
    def _best_combination(
        self,
        q: Constr,
        r: Constr,
        q_coeffs: Dict[Var, float],
        r_coeffs: Dict[Var, float],
        tol: float,
        ratio_tol: float,
        min_pivot: float,
        max_s: float,
        min_gain: int,
    ) -> Optional[Tuple[float, Dict[Var, float], float, int]]:
        """
        Try every ratio s = A_rj / A_qj for j in supp(q) ∩ supp(r), pick the
        one maximizing |S| - |U|, and return the resulting new row for r.

        Returns (s, new_coeffs_for_r, new_rhs_for_r, gain) or None.
        """
        intersection = [v for v in q_coeffs if v in r_coeffs]
        if not intersection:
            return None

        # |U| depends only on q and r, not on s.
        u_size = sum(1 for v in q_coeffs if v not in r_coeffs)

        # Bucket intersection columns by their ratio A_rj / A_qj. All columns
        # in the same bucket cancel simultaneously for that s. Floats are
        # quantized to integer keys so tolerance-based equality holds.
        ratio_buckets: Dict[int, List[Var]] = defaultdict(list)
        for v in intersection:
            a_q = q_coeffs[v]
            if abs(a_q) < min_pivot:
                continue
            ratio = r_coeffs[v] / a_q
            key = self._quantize(ratio, ratio_tol)
            ratio_buckets[key].append(v)

        if not ratio_buckets:
            return None

        # Pick the bucket with the largest |S|. Ties broken by preferring an
        # |s| closer to 1 (numerically safest).
        def bucket_score(k: int) -> Tuple[int, float]:
            return (len(ratio_buckets[k]), -abs(math.log1p(abs(k * ratio_tol))))

        best_key = max(ratio_buckets, key=bucket_score)
        s_columns = ratio_buckets[best_key]
        s_size = len(s_columns)

        if s_size - u_size < min_gain:
            return None

        # Recompute s exactly from a representative column to avoid drift
        # from the quantization.
        rep = s_columns[0]
        s = r_coeffs[rep] / q_coeffs[rep]

        if not math.isfinite(s) or abs(s) > max_s or abs(s) < min_pivot:
            return None

        # Build the new row: r_new[j] = r[j] - s * q[j] over supp(q) ∪ supp(r).
        new_coeffs: Dict[Var, float] = {}

        # Columns already in r (yields T and S; S drops out, T stays)
        for v, a_r in r_coeffs.items():
            a_q = q_coeffs.get(v, 0.0)
            coeff = a_r - s * a_q
            if abs(coeff) > tol:
                new_coeffs[v] = coeff

        # Columns in q but not in r (this is U, introduced into r)
        for v, a_q in q_coeffs.items():
            if v in r_coeffs:
                continue
            coeff = -s * a_q
            if abs(coeff) > tol:
                new_coeffs[v] = coeff

        # Recompute gain against reality: near-zero coefficients in T may
        # collapse (good) or bucketing may have over-counted (bad).
        actual_gain = len(r_coeffs) - len(new_coeffs)
        if actual_gain < min_gain:
            return None

        new_rhs = float(r.rhs) - s * float(q.rhs)
        return s, new_coeffs, new_rhs, actual_gain

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_supports(
        self,
        tol: float,
    ) -> Tuple[Dict[Constr, Dict[Var, float]], Dict[Var, List[Constr]]]:
        """Build row->{var: coeff} and var->[rows] indices in one pass."""
        row_support: Dict[Constr, Dict[Var, float]] = {}
        col_support: Dict[Var, List[Constr]] = defaultdict(list)

        for c in self.mip.constrs:
            expr: LinExpr = c.expr
            coeffs: Dict[Var, float] = {}
            for var, coef in expr.expr.items():
                cf = float(coef)
                if abs(cf) > tol:
                    coeffs[var] = cf
                    col_support[var].append(c)
            row_support[c] = coeffs

        return row_support, col_support

    @staticmethod
    def _is_equality(c: Constr) -> bool:
        return c.expr.sense == "="

    @staticmethod
    def _quantize(x: float, ratio_tol: float) -> int:
        """Bucket a float ratio for tolerance-aware grouping."""
        if x == 0.0:
            return 0
        return int(round(x / ratio_tol))
