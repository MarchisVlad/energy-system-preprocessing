from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mip
from mip import Constr, LinExpr, Var

from src.presolvers.base import StaticPresolvingAlgorithm


class Sparsify(StaticPresolvingAlgorithm):
    """
    Non-zero cancellation ("sparsify") presolver.

    For an equality row q and another row r, find a scalar s such that
    replacing row r by (row_r - s * row_q) reduces the total number of
    non-zeros in the constraint matrix.

    Reference: Achterberg et al., "Presolve Reductions in Mixed Integer
    Programming" (2019); PaPILO's Sparsify presolver.
    """

    @property
    def name(self) -> str:
        return "Sparsify"

    @property
    def slug(self) -> str:
        return "sparsify_static"

    def _run(self, model) -> None:
        self.mip: mip.Model = model.model
        self.presolve()

    def presolve(
        self,
        max_rounds: int = 1,
        tol: float = 1e-9,
        ratio_tol: float = 1e-9,
        min_pivot: float = 1e-7,
        max_s: float = 1e6,
        min_gain: int = 1,
        **kwargs,
    ) -> None:
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

    def _run_pass(
        self,
        tol: float,
        ratio_tol: float,
        min_pivot: float,
        max_s: float,
        min_gain: int,
    ) -> int:
        row_support, col_support = self._build_supports(tol=tol)

        constraints: List[Constr] = list(self.mip.constrs)
        row_index: Dict[Constr, int] = {c: i for i, c in enumerate(constraints)}

        eq_rows: List[Constr] = sorted(
            (c for c in constraints if self._is_equality(c)),
            key=lambda c: len(row_support[c]),
        )

        modified_rows: set = set()
        changes: List[Tuple[int, str, Dict[Var, float], float, str]] = []

        for q in eq_rows:
            q_coeffs = row_support[q]
            if len(q_coeffs) < 2:
                continue

            pivot_var = min(q_coeffs, key=lambda v: len(col_support[v]))
            candidates = {
                r for r in col_support[pivot_var]
                if r is not q and r not in modified_rows
            }

            for r in candidates:
                if r in modified_rows:
                    continue
                result = self._best_combination(
                    q, r, q_coeffs, row_support[r],
                    tol=tol, ratio_tol=ratio_tol,
                    min_pivot=min_pivot, max_s=max_s, min_gain=min_gain,
                )
                if result is None:
                    continue
                _s, new_coeffs, new_rhs, _gain = result
                changes.append((row_index[r], r.name, new_coeffs, new_rhs, r.expr.sense))
                modified_rows.add(r)

        if not changes:
            return 0

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
        intersection = [v for v in q_coeffs if v in r_coeffs]
        if not intersection:
            return None

        u_size = sum(1 for v in q_coeffs if v not in r_coeffs)

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

        def bucket_score(k: int) -> Tuple[int, float]:
            return (len(ratio_buckets[k]), -abs(math.log1p(abs(k * ratio_tol))))

        best_key = max(ratio_buckets, key=bucket_score)
        s_columns = ratio_buckets[best_key]
        s_size = len(s_columns)

        if s_size - u_size < min_gain:
            return None

        rep = s_columns[0]
        s = r_coeffs[rep] / q_coeffs[rep]

        if not math.isfinite(s) or abs(s) > max_s or abs(s) < min_pivot:
            return None

        new_coeffs: Dict[Var, float] = {}
        for v, a_r in r_coeffs.items():
            a_q = q_coeffs.get(v, 0.0)
            coeff = a_r - s * a_q
            if abs(coeff) > tol:
                new_coeffs[v] = coeff

        for v, a_q in q_coeffs.items():
            if v in r_coeffs:
                continue
            coeff = -s * a_q
            if abs(coeff) > tol:
                new_coeffs[v] = coeff

        actual_gain = len(r_coeffs) - len(new_coeffs)
        if actual_gain < min_gain:
            return None

        new_rhs = float(r.rhs) - s * float(q.rhs)
        return s, new_coeffs, new_rhs, actual_gain

    def _build_supports(
        self, tol: float
    ) -> Tuple[Dict[Constr, Dict[Var, float]], Dict[Var, List[Constr]]]:
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
        if x == 0.0:
            return 0
        return int(round(x / ratio_tol))
