from abc import ABC, abstractmethod
from enum import Enum

import mip
import scipy.sparse as sp

from src.core.model import Model
from src.core.presolving import Presolver, PresolvingMethod


class PresolveAlgorithm(ABC):
    """
    Abstract umbrella for all presolving algorithms.
    """

    name: str

    @abstractmethod
    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def presolve(self, **kwargs):
        """
        Analyze model and perform a presolve step, returning the original model.
        """
        pass


def presolve(
    model: Model,
    method: PresolvingMethod,
    presolver: Presolver = Presolver.Static,
    **kwargs,
) -> Model:
    """
    Apply a presolve method to the model, updating the model in place.

    Parameters
    ----------
    method : PresolvingMethod
        The presolve method to apply.
    presolver : Presolver.Static
        The presolver instance to use for applying the method.
    **kwargs
        Additional keyword arguments to pass to the presolver method.
    """

    if method == "None":
        return model

    elif method == PresolvingMethod.CoeffTightening:
        from src.presolvers.techniques.coefficient_strengthening import (
            CoefficientStrengthening,
        )

        algorithm = CoefficientStrengthening(model)
        algorithm.presolve(**kwargs)
        return model

    else:
        raise ValueError(f"Unknown/Unimplemented presolve method: {method}")
