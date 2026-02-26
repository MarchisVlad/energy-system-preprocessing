import math
from abc import ABC, abstractmethod
from enum import Enum

import mip
import numpy as np
import scipy.sparse as sp

from src.core.block import BlockStructure
from src.core.model import Model
from src.presolvers.algorithm import PresolveAlgorithm
from src.core.presolving import PresolveStatus


class CoefficientStrengthening(PresolveAlgorithm):
    """
    Coefficient strengthening/tightening presolver for MIP problems.

    For each row of the form a^T x <= b (or >= b), if a variable x_j is integer
    and its coefficient |a_j| > (maxactivity - b), we can tighten a_j down to
    (maxactivity - b) and adjust the RHS accordingly, without changing the
    feasible integer set.

    Reference: CoefficientStrengthening.hpp from PaPILO
    """

    name: str = "coefftightening"

    def __init__(self, model: Model):
        self.model = model

    def presolve(self, **kwargs) -> sp.spmatrix:
        """
        Analyze matrix A and return a presolved matrix.

        Iterates over all rows and applies coefficient tightening where possible.
        Modifies model.A, model.lhs, model.rhs in place and returns the new A.
        """
        A = self.model.A  # scipy sparse matrix (num_rows x num_cols), CSR preferred
        if not sp.isspmatrix_csr(A):
            A = A.tocsr()

        lhs = np.array(self.model.lhs,
                       dtype=float)  # lower bounds on rows (or -inf)
        rhs = np.array(self.model.rhs,
                       dtype=float)  # upper bounds on rows (or +inf)
        lb = np.array(self.model.lb, dtype=float)  # variable lower bounds
        ub = np.array(self.model.ub, dtype=float)  # variable upper bounds

        # Boolean arrays: which variables are integer or implied integer
        is_integer = np.array(self.model.is_integer,
                              dtype=bool)  # shape (num_cols,)

        num_rows, num_cols = A.shape
        A_lil = A.tolil()  # use LIL for efficient element-wise modification

        overall_status = PresolveStatus.kUnchanged

        for row_idx in range(num_rows):
            status = self._tighten_row(row_idx, A_lil, lhs, rhs, lb, ub,
                                       is_integer)
            if status == PresolveStatus.kReduced:
                overall_status = PresolveStatus.kReduced

        self.model.A = A_lil.tocsr()
        self.model.lhs = lhs.tolist()
        self.model.rhs = rhs.tolist()
        return self.model.A

    def _tighten_row(
        self,
        row_idx: int,
        A_lil: sp.lil_matrix,
        lhs: np.ndarray,
        rhs: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        is_integer: np.ndarray,
    ) -> PresolveStatus:
        """
        Apply coefficient tightening to a single row.

        The logic mirrors perform_coefficient_tightening() in the C++ source:
          1. Skip equality rows (both lhs and rhs finite) and rows with <= 1 nonzero.
          2. Normalise to a^T x <= b form (multiply by -1 if it is a >= row).
          3. Compute maxactivity of the normalised row.
          4. Compute newabscoef = maxactivity - b  (the tightened absolute value).
          5. For every integer variable whose |coef| > newabscoef, replace the
             coefficient with ±newabscoef and adjust b to compensate.
          6. Write back the (possibly scaled) changes to A_lil, lhs/rhs.
        """
        row_data = A_lil.getrowview(row_idx).toarray().flatten()
        nonzero_cols = np.nonzero(row_data)[0]

        row_lhs = lhs[row_idx]
        row_rhs = rhs[row_idx]

        lhs_inf = math.isinf(row_lhs) and row_lhs < 0  # lhs == -inf
        rhs_inf = math.isinf(row_rhs) and row_rhs > 0  # rhs == +inf

        # Skip equality rows (both sides finite) — can't tighten these
        if not lhs_inf and not rhs_inf:
            return PresolveStatus.kUnchanged

        # Skip rows with 0 or 1 nonzero (nothing useful to tighten)
        if len(nonzero_cols) <= 1:
            return PresolveStatus.kUnchanged

        # ------------------------------------------------------------------
        # Normalise to a^T x <= b.  scale = -1 means original row was >= .
        # ------------------------------------------------------------------
        if not lhs_inf:
            # Row is:  lhs <= a^T x  →  (-a)^T x <= -lhs
            # maxactivity of normalised row = -minactivity of original row
            min_act, min_act_ok = self._compute_min_activity(
                row_data, nonzero_cols, lb, ub)
            if not min_act_ok:
                return PresolveStatus.kUnchanged
            maxact = -min_act
            b = -row_lhs
            scale = -1
        else:
            # Row is:  a^T x <= rhs  (already normalised)
            max_act, max_act_ok = self._compute_max_activity(
                row_data, nonzero_cols, lb, ub)
            if not max_act_ok:
                return PresolveStatus.kUnchanged
            maxact = max_act
            b = row_rhs
            scale = 1

        # The tightened absolute coefficient value
        newabscoef = maxact - b
        if abs(newabscoef) < 1e-10:
            newabscoef = 0.0
        else:
            ceil_val = math.ceil(newabscoef)
            if abs(newabscoef - ceil_val) < 1e-6:
                newabscoef = ceil_val

        # If newabscoef <= 0 every integer coefficient is already at most 0,
        # nothing to tighten (and the row might already be redundant).
        if newabscoef < 0:
            return PresolveStatus.kUnchanged

        # ------------------------------------------------------------------
        # Collect integer variables whose |coef| (in normalised form) can be
        # tightened, i.e. |scale * a_j| > newabscoef.
        # ------------------------------------------------------------------
        to_tighten = []  # list of (normalised_coef, col_idx)

        for col in nonzero_cols:
            if not is_integer[col]:
                continue
            if lb[col] == ub[col]:
                continue

            norm_coef = row_data[col] * scale  # coef in normalised a^T x <= b
            if abs(norm_coef) <= newabscoef + 1e-10:
                continue  # already tight enough

            to_tighten.append([norm_coef, col])

        if not to_tighten:
            return PresolveStatus.kUnchanged

        # Sanity: maxact must exceed b for the tightening to make sense
        if not (maxact > b + 1e-10):
            return PresolveStatus.kUnchanged

        # ------------------------------------------------------------------
        # Adjust b and replace each qualifying coefficient with ±newabscoef.
        # The RHS adjustment preserves the set of feasible integer points:
        #
        #   positive coef:  b  +=  (newabscoef - a_j) * ub[col]
        #   negative coef:  b  -=  (newabscoef + a_j) * lb[col]   (a_j < 0)
        # ------------------------------------------------------------------
        for entry in to_tighten:
            norm_coef, col = entry
            if norm_coef < 0:
                assert not math.isinf(lb[col]), (
                    f"Variable {col} has -inf lower bound but negative coef")
                b -= (newabscoef + norm_coef) * lb[col]
                entry[0] = -newabscoef
            else:
                assert not math.isinf(ub[col]), (
                    f"Variable {col} has +inf upper bound but positive coef")
                b += (newabscoef - norm_coef) * ub[col]
                entry[0] = newabscoef

        # ------------------------------------------------------------------
        # Write changes back (un-normalise with scale).
        # ------------------------------------------------------------------
        for norm_coef, col in to_tighten:
            A_lil[row_idx, col] = norm_coef * scale  # original orientation

        if scale == -1:
            lhs[row_idx] = -b
        else:
            rhs[row_idx] = b

        return PresolveStatus.kReduced

    # ------------------------------------------------------------------
    # Activity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_activity(
        row: np.ndarray,
        nonzero_cols: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """
        Compute max activity = sum_j { a_j * ub[j] if a_j > 0 else a_j * lb[j] }.
        Returns (value, is_finite).  Returns (None, False) if any required bound
        is infinite.
        """
        maxact = 0.0
        for col in nonzero_cols:
            coef = row[col]
            if coef > 0:
                if math.isinf(ub[col]):
                    return None, False
                maxact += coef * ub[col]
            else:
                if math.isinf(lb[col]):
                    return None, False
                maxact += coef * lb[col]
        return maxact, True

    @staticmethod
    def _compute_min_activity(
        row: np.ndarray,
        nonzero_cols: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """
        Compute min activity = sum_j { a_j * lb[j] if a_j > 0 else a_j * ub[j] }.
        Returns (value, is_finite).
        """
        minact = 0.0
        for col in nonzero_cols:
            coef = row[col]
            if coef > 0:
                if math.isinf(lb[col]):
                    return None, False
                minact += coef * lb[col]
            else:
                if math.isinf(ub[col]):
                    return None, False
                minact += coef * ub[col]
        return minact, True
