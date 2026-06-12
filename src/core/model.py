"""Module for manipulating and interacting with different optimisation model providers.

Notes
-----
    `Sense`, `Problem` and `FileFormat` are implemented in the same way as the
    `gamspy` API <https://gamspy.readthedocs.io/en/latest/reference/index.html>
    but are duplicated to represent limitations of the current implementation
    and facilitate translation between mip.Model and gp.Model specific methods.
"""

import contextlib
import ctypes
import os
import sys
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gamspy as gp
import mip
import numpy as np
import scipy.sparse as sp

from src.core.block import BlockStructure
from src.core.matrix import Matrix, MatrixFormat
from src.core.presolving import Presolver, PresolvingMethod

# Load libc once
try:
    _libc = ctypes.CDLL(None)  # Linux/macOS
except OSError:
    _libc = ctypes.CDLL("msvcrt")  # Windows fallback


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress C-level stdout output (e.g., from CBC/COIN solver)."""
    sys.stdout.flush()
    _libc.fflush(None)  # flush all C stdio streams

    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        _libc.fflush(None)  # flush CBC's buffered output into /dev/null
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


class Sense(Enum):
    """An enumeration for sense types"""

    MIN = "MIN"
    """Minimize the objective."""

    MAX = "MAX"
    """Maximize the objective."""

    FEASIBILITY = "FEASIBILITY"
    """Assess feasibility."""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Problem(Enum):
    """An enumeration for supported problem types"""

    LP = "LP"
    "Linear Programming"

    MIP = "MIP"
    """Mixed Integer Programming"""

    NLP = "NLP"
    """Non-Linear Programming"""

    QCP = "QCP"
    """Quadratically Constrained Programs"""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class FileFormat(Enum):
    """An enumeration for file format types"""

    CPLEXLP = "cplex.lp"
    """CPLEX LP format."""

    CPLEXMPS = "cplex.mps"
    """CPLEX MPS format."""

    GAMSDict = "dict.txt"
    """GAMS dictionary format."""

    GAMSDictMap = "dictmap.gdx"
    """GAMS dictionary map format."""

    GDXJacobian = "jacobian.gdx"
    """GDX file with model data incl. Jacobian and Hessian evaluated at current point."""

    GDXUnload = "execute_unload.gdx"
    """GDX file from unload execution prior to solve"""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Model:
    """Class representing an optimisation model.

    Attributes
    ----------
        model : Union[mip.Model, gp.Model]
            An existing model object to wrap.
        A : sp.coo_matrix
            Sparse representation of the model's constrain matrices.
        type : Problem
            Sense of optimisation problem (minimisation/maximisation).
            Default is ProblemType.MIN.
        sense : Sense
            Sense of optimisation problem (minimisation/maximisation).
            Default is ProblemType.MIN.
        blocks : BlockStructure, optional
            Optional decomposition blocks.
        presolves : list[PresolvingMethod], optional
            Optional presolve steps.
    """

    def __init__(
        self,
        model: Union[mip.Model, gp.Model] | None = None,
        path: str | None = "",
        format: FileFormat | None = None,
        A: sp.spmatrix | None = None,
        problem: Problem | None = Problem.MIP,
        sense: Sense | None = Sense.MIN,
        blocks: BlockStructure | None = None,
        presolves: list[PresolvingMethod] | None = [],
    ):
        """
        Construct a Model instance.

        Parameters
        ----------
        model : mip.Model or gp.Model, optional
            An existing model object to wrap.
        path : str, optional
            Path to a model file to load.
        format : ModelFormat, optional
            The format of the model (MPS, GAMSPY, etc.). Required if
            loading from file or wrapping an existing model.
        A : sp.coo_matrix, optional
            Constraint matrix representing the model.
        problem : Problem, optional
            Class of optimization problem (MIP, LP, etc.).
            Default is ProblemClass.MIP.
        sense : Sense, optional
            Type of optimization problem (minimization/maximization).
            Default is ProblemType.MIN.
        blocks : BlockStructure, optional
            Optional decomposition blocks.
        presolves : list[PresolvingMethod], optional
            Optional presolve steps.
        """

        self.model = model
        self.path = "" if path is None else path
        self.format = format
        self._A = A
        self.problem = problem
        self.sense = sense
        self.blocks = blocks
        self.presolves = [] if presolves is None else presolves

        # Initialise from path if specified.
        if self.path:
            if format is None:
                if self.path.endswith(".mps"):
                    format = FileFormat.CPLEXMPS
                elif self.path.endswith(".lp"):
                    format = FileFormat.CPLEXLP
                elif self.path.endswith(".gdx"):
                    format = FileFormat.GDXUnload
                else:
                    raise ValueError(
                        "File format must be specified when loading from path")

            if format == FileFormat.CPLEXMPS:
                self.model = mip.Model()
                with _suppress_stdout():
                    self.model.read(path=self.path)

            else:
                # TODO: Handle initialisation for other formats.
                pass

        # Wrap around pre-exisiting models.
        if self.model is not None:
            self._A = self._extract_matrix()
            self._integers = self._extract_integers()
            lo, up = self._extract_bounds()
            self._lo = lo
            self._up = up

    @property
    def container(self) -> gp.Container:
        if not hasattr(self, "_container") or self._container is None:
            raise AttributeError(
                "No GDX container loaded. Initialise with FileFormat.GDXUnload."
            )
        return self._container

    @property
    def A(self, without_objective=False) -> sp.spmatrix:
        if self._A is None:
            raise ValueError("Model matrix A is not set.")

        # return self._A[1:][1:] if without_objective else self._A
        return self._A

    def integers(self, col_perm: np.ndarray = None) -> np.ndarray:
        """Return the 0/1 integer-variable indicator vector.

        Parameters
        ----------
        col_perm : np.ndarray, optional
            Column permutation (as returned by apply_reordering or
            detect_block_structure). When provided, returns the integers
            vector reindexed to match the permuted column order.
        """
        if self._integers is None:
            raise ValueError("Model integers are not set.")

        if col_perm is not None:
            return self._integers[col_perm]

        return self._integers

    def update_matrix(self):
        if self.model is not None:
            self._A = self._extract_matrix()
        else:
            raise RuntimeError(
                "Tried updating model matrix but self.model is not set. "
                "Probably caused by initialisation with just the matrix component."
            )

    def convert(self, type: type[mip.Model | gp.Model]):
        """
        Converts the internal ``model`` attribute to the specified format.

        Currently supports conversion between :class:`mip.Model` and :class:`gp.Model`.

        Parameters
        ----------
        format : ModelFormat
            The target model format.
        """

        if self.model is None:

            raise RuntimeError(
                "Attempted to convert a model but the "
                "self.model attribute is not set. Probably caused by "
                "initialisation with just the matrix component.")

        if isinstance(self.model, type):
            warnings.warn(f"Model is already in the desired format: {type}",
                          UserWarning)

        conversion_map = {
            mip.Model: lambda m: m._to_mip(),
            gp.Model: lambda m: m._to_gamspy(),
        }

        conversion_map[type](self)

    def _to_mip(self):
        if isinstance(self.model, gp.Model):
            with tempfile.TemporaryDirectory() as tempdir:
                options = gp.ConvertOptions(GAMSObjVar="obj")
                self.model.convert(tempdir,
                                   gp.FileFormat.FixedMPS,
                                   options=options)
                self.model = mip.Model()
                self.model.read(
                    path=str(Path(tempdir) / gp.FileFormat.FixedMPS.value))
                self._A = self._extract_matrix()
        else:
            # TODO: implement conversions to other available types.
            pass

    def _to_gamspy(self):
        # TODO: implement conversion to gamspy
        pass

    def _extract_matrix(self) -> sp.spmatrix:
        """
        Function that returns the constraint matrix of the model.

        Returns
        ------
        Matrix
            A sparse matrix representation of the constraint matrix.
        """
        if isinstance(self.model, mip.Model):

            n_rows = len(self.model.constrs)
            n_cols = len(self.model.vars)

            data, rows, cols = [], [], []

            for i, constr in enumerate(self.model.constrs):
                expr = constr.expr
                for var, coeff in expr.expr.items():
                    rows.append(i)
                    cols.append(var.idx)
                    data.append(1 if coeff != 0 else 0)

            return sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

        elif isinstance(self.model, gp.Model):
            # Convert to MPS in a temp dir to extract the matrix without
            # mutating self.model (which must stay as gp.Model).
            with tempfile.TemporaryDirectory() as tempdir:
                options = gp.ConvertOptions(GAMSObjVar="obj")
                self.model.convert(tempdir,
                                   gp.FileFormat.FixedMPS,
                                   options=options)
                tmp = mip.Model()
                with _suppress_stdout():
                    tmp.read(
                        path=str(Path(tempdir) / gp.FileFormat.FixedMPS.value))
                n_rows = len(tmp.constrs)
                n_cols = len(tmp.vars)
                data, rows, cols = [], [], []
                for i, constr in enumerate(tmp.constrs):
                    for var, coeff in constr.expr.expr.items():
                        rows.append(i)
                        cols.append(var.idx)
                        data.append(1 if coeff != 0 else 0)
                return sp.coo_matrix((data, (rows, cols)),
                                     shape=(n_rows, n_cols))

        raise TypeError(f"Unsupported model type: {type(self.model)!r}")

    def _extract_integers(self) -> np.ndarray:
        """
        Return a 0/1 vector marking integer-restricted columns
        in GAMS style.

        Returns
        -------
        np.ndarray
            1 for integer/binary columns, 0 otherwise.
        """
        if isinstance(self.model, mip.Model):
            return np.fromiter(
                (1 if var.var_type in (mip.BINARY, mip.INTEGER) else 0
                 for var in self.model.vars),
                dtype=np.int8,
                count=len(self.model.vars),
            )

        elif isinstance(self.model, gp.Model):
            # Straight GAMSPy version if you already use a flattened
            # variable ordering elsewhere.
            vars_ = self.model.container.getVariables()
            integer_like = {"binary", "integer", "semiint"}

            return np.fromiter(
                (1 if str(var.type).lower() in integer_like else 0
                 for var in vars_),
                dtype=np.int8,
                count=len(vars_),
            )

        raise TypeError(f"Unsupported model type: {type(self.model)!r}")

    def _extract_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return lower and upper bounds of the model variables.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lo, up) arrays aligned with the variable ordering.
        """
        if isinstance(self.model, mip.Model):
            n_cols = len(self.model.vars)
            lo = np.empty(n_cols, dtype=float)
            up = np.empty(n_cols, dtype=float)

            for j, var in enumerate(self.model.vars):
                # Strict GAMS-style handling for binaries:
                # binary variables are 0/1 variables.
                if var.var_type == mip.BINARY:
                    lo[j] = 0.0
                    up[j] = 1.0
                else:
                    lo[j] = float(var.lb)
                    up[j] = float(var.ub)

            return lo, up

        elif isinstance(self.model, gp.Model):
            # This assumes your GAMSPy column ordering is exactly the order
            # returned by container.getVariables(). If you flatten indexed
            # variables elsewhere, use the same flattening here too.
            vars_ = self.model.container.getVariables()

            lo = np.empty(len(vars_), dtype=float)
            up = np.empty(len(vars_), dtype=float)

            for j, var in enumerate(vars_):
                vtype = str(var.type).lower()

                # If records are present, use explicit lower/upper values.
                # Otherwise fall back to GAMS defaults by type.
                if var.records is not None:
                    # For scalar variables this is one row; for indexed symbols
                    # you probably want your own flattening logic instead.
                    lo[j] = float(var.records["lower"].iloc[0])
                    up[j] = float(var.records["upper"].iloc[0])
                else:
                    if vtype == "binary":
                        lo[j], up[j] = 0.0, 1.0
                    elif vtype == "integer":
                        lo[j], up[j] = 0.0, np.inf
                    elif vtype == "semiint":
                        lo[j], up[j] = 1.0, np.inf
                    elif vtype == "positive":
                        lo[j], up[j] = 0.0, np.inf
                    elif vtype == "negative":
                        lo[j], up[j] = -np.inf, 0.0
                    elif vtype == "semicont":
                        lo[j], up[j] = 1.0, np.inf
                    else:  # free, sos1, sos2, etc. -> use your preferred convention
                        lo[j], up[j] = -np.inf, np.inf

            return lo, up

        raise TypeError(f"Unsupported model type: {type(self.model)!r}")


class ModelHistory:
    """Wrapper for interacting with a model series of presolves.

    Basically a list of models with logging features.

    Attributes
    ----------
    states : List[Model]
        The history of the current presolves applied.

    presolves : List[PresolveMethod]
        A copy of the presolve methods applied to the model.
        Also stored by the Model object.

    current_index : int
        The current index. Used by the GUI.

    logs : List[str]
        Logging information used for debugging.
    """

    def __init__(self, model: Model):
        """Initalise a new model history for a pre-exisiting Model.

        Parameters
        ----------
        model : Model
            The initial model. `states` and `current_index` will be
            initialised using data of the model.

        Returns
        -------
            A `ModelHistory` object.
        """
        self.states: list[Tuple[Optional[PresolvingMethod],
                                Model]] = [(None, model)]
        self.current_index: int = len(self.states) - 1

    def add_state(self, step: Optional[PresolvingMethod], model: Model):
        self.states = self.states[:self.current_index + 1]
        self.states.append((step, model))
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
