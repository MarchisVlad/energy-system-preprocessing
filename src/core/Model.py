from enum import Enum
from typing import List, Optional, Tuple

import gamspy as gp
import mip
import numpy as np
import scipy.sparse as sp

from .Presolving import PresolvingMethod


class ProblemType(Enum):
    MIN = 1
    MAX = 2


class ProblemClass(Enum):
    LP = 1
    MIP = 2
    QP = 3


class ModelFormat(Enum):
    MPS = 1
    GMS = 2
    GDX = 3
    LP = 4


class Model:

    def __init__(
        self,
        A: sp.coo_matrix | None = None,
        problem_type: ProblemType | None = ProblemType.MIN,
        problem_class: ProblemClass | None = ProblemClass.MIP,
        model_format: ModelFormat | None = ModelFormat.MPS,
        blocks: list | None = None,
        presolves: list | None = None,
    ):
        self.A = A
        self.problem_type = problem_type
        self.problem_class = problem_class
        self.model_format = model_format
        self.blocks = blocks
        self.presolves = presolves

    def __init__(self, path, model_format=ModelFormat.MPS):
        self.model_format = model_format

        if model_format == ModelFormat.MPS:
            # Read MPS file using python-mip
            self.model = mip.Model()
            self.model.read(path=path)

        elif model_format == ModelFormat.GMS:
            # TODO: Handle initialisation for GMS formats.
            pass

        self.A = self._extract_matrix(self.model)

    def __post_init__(self):
        if (isinstance(self.model, mip.Model) and
                not self.model_format is ModelFormat.MPS):
            raise TypeError(
                "Models must have their format specified: attempted to " \
                "construct a mip.Model without setting the model_format " \
                "parameter.")
        if (isinstance(self.model, gp.Model) and
                not self.model_format is ModelFormat.GMS):
            raise TypeError(
                "Models must have their format specified: attempted to " \
                "construct a gp.Model without setting the model_format " \
                "parameter."
            )

    def _extract_matrix(self, model) -> sp.coo_matrix:
        if self.model_format == ModelFormat.MPS:

            n_rows = len(model.constrs)
            n_cols = len(model.vars)

            data, rows, cols = [], [], []

            for i, constr in enumerate(model.constrs):
                expr = constr.expr
                for var, coeff in expr.expr.items():
                    rows.append(i)
                    cols.append(var.idx)
                    data.append(coeff)

            return sp.coo_matrix((data, (rows, cols)),
                                 shape=(n_rows, n_cols)).tocsr()

        elif self.model_format == ModelFormat.GMS:
            # TODO: Matrix extraction for GMS formats.
            pass

    def _get_col_ordering(self, row_perm):
        """Get column ordering that follows row ordering"""
        # Simple heuristic: order columns by first row they appear in
        A_reordered = self.A[row_perm]

        col_first_row = np.zeros(self.n_cols, dtype=int)
        for col in range(self.n_cols):
            col_data = A_reordered.getcol(col)
            rows = col_data.nonzero()[0]
            if len(rows) > 0:
                col_first_row[col] = rows[0]
            else:
                col_first_row[col] = self.n_rows  # Put empty columns at end

        col_perm = np.argsort(col_first_row)
        return col_perm


class ModelHistory:

    def __init__(self, original: Model):
        self.original = original
        self.states: List[Tuple[Optional[PresolvingMethod],
                                sp.coo_matrix]] = [(None, original.A)]
        self.current_index: int = 0

    def add_state(self, step: Optional[PresolvingMethod], A: sp.coo_matrix):
        self.states = self.states[:self.current_index + 1]
        self.states.append((step, A))
        self.current_index = len(self.states) - 1

    def get_current_state(self):
        if 0 <= self.current_index < len(self.states):
            return self.states[self.current_index]
        return None

    def revert_to_index(self, index: int) -> bool:
        if 0 <= index < len(self.states):
            self.current_index = index
            return True
        return False

    def get_state_at_index(self, index: int):
        if 0 <= index < len(self.states):
            return self.states[index]
        return None

    def get_history_summary(self) -> List[str]:
        summaries = []
        for step, _ in self.states:
            summaries.append(step.name if step else "Original Model")
        return summaries
