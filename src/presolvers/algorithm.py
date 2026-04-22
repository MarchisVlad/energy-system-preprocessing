from abc import ABC, abstractmethod
from enum import Enum

import mip
import scipy.sparse as sp

from src.core.model import Model
from src.core.presolving import PresolvingMethod


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


def _resolve_algorithm(method: PresolvingMethod) -> type[PresolveAlgorithm]:
    """Return the algorithm class for *method*, importing it lazily."""
    if method == PresolvingMethod.CoeffTightening:
        from src.presolvers.techniques.coefficient_strengthening import \
            CoefficientStrengthening
        return CoefficientStrengthening
    if method == PresolvingMethod.Propagation:
        from src.presolvers.techniques.constraint_propagation import \
            ConstraintPropagation
        return ConstraintPropagation
    if method == PresolvingMethod.FixContinuous:
        from src.presolvers.techniques.fix_continuous import FixContinuous
        return FixContinuous
    if method == PresolvingMethod.DualFix:
        from src.presolvers.techniques.dual_fix import DualFix
        return DualFix
    if method == PresolvingMethod.Sparsify:
        from src.presolvers.techniques.sparsify import Sparsify
        return Sparsify
    raise ValueError(f"Unknown/Unimplemented presolve method: {method}")


def presolve(
    model: Model,
    method: PresolvingMethod,
    **kwargs,
) -> Model:
    """
    Apply a presolve method to the model, updating the model in place.

    Parameters
    ----------
    method : PresolvingMethod
        The presolve method to apply.
    **kwargs
        Additional keyword arguments to pass to the presolver method.
    """
    if method == "None":
        return model

    _resolve_algorithm(method)(model).presolve(**kwargs)
    return model
