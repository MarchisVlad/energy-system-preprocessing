"""Module for manipulating and interacting with different optimisation model providers.

Notes
-----
    `Sense`, `Problem` and `FileFormat` are implemented in the same way as the 
    `gamspy` API <https://gamspy.readthedocs.io/en/latest/reference/index.html>
    but are duplicated to represent limitations of the current implementation 
    and facilitate translation between mip.Model and gp.Model specific methods.
"""

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
from src.core.presolving import PresolvingMethod


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
        path: str | None = None,
        format: FileFormat | None = None,
        A: Matrix | None = None,
        problem: Problem | None = Problem.MIP,
        sense: Sense | None = Sense.MIN,
        blocks: BlockStructure | None = None,
        presolves: list[PresolvingMethod] | None = None,
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
        self.path = path
        self.format = format
        self._A = A
        self.problem = problem
        self.sense = sense
        self.blocks = blocks
        self.presolves = presolves

        # Initialise from path if specified.
        if self.path is not None:
            if format is None:
                raise ValueError(
                    'File format must be specified when loading from path')

            if format == FileFormat.CPLEXMPS:
                self.model = mip.Model()
                self.model.read(path=path)

            elif format == None:
                # TODO: Handle initialisation for other formats.
                pass

        # Wrap around pre-exisiting models.
        if self.model is not None:
            self._A = self._extract_matrix()

    @property
    def A(self, without_objective=False):
        return self._A[1:][1:] if without_objective else self._A

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

            raise RuntimeError("Attempted to convert a model but the " \
            "self.model attribute is not set. Probably caused by " \
            "initialisation with just the matrix component.")

        if isinstance(self.model, type):
            warnings.warn(f'Model is already in the desired format: {type}',
                          UserWarning)

        conversion_map = {
            mip.Model: lambda m: m._to_mip(),
            gp.Model: lambda m: m._to_gamspy()
        }

        conversion_map[type](self)

    def _to_mip(self):
        if isinstance(self.model, gp.Model):
            with tempfile.TemporaryDirectory as tempdir:
                options = gp.ConvertOptions(GAMSObjVar='obj')
                self.model.convert(tempdir,
                                   gp.FileFormat.FixedMPS,
                                   options=options)
                self.model = mip.Model()
                self.model.read(path=(Path(tempdir) / 'fixed.MPS'))
                self.A = self._extract_matrix(self.model)
        else:
            # TODO: implement conversions to other available types.
            pass

    def _to_gamspy(self):
        # TODO: implement conversion to gamspy
        pass

    def _extract_matrix(self) -> Matrix:
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

            return Matrix((data, (rows, cols)),
                          shape=(n_rows, n_cols)).convert(MatrixFormat.CSR)

        elif isinstance(self.model, gp.Model):
            # TODO: Matrix extraction for GMS formats.
            pass


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
        self.states: list[Model] = {model}
        self.presolves = model.presolves
        self.current_index: int = len(model.presolves)

    def add_state(self, step: Optional[PresolvingMethod], A: Matrix):
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
