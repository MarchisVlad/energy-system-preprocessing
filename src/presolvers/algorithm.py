from abc import ABC, abstractmethod
from enum import Enum

import mip
import scipy.sparse as sp

from src.core.block import BlockStructure
from src.core.model import Model


class PresolveAlgorithm(ABC):
    """
    Abstract umbrella for all block detection algorithms.
    """

    name: str

    @abstractmethod
    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def presolve(self, **kwargs) -> BlockStructure:
        """
        Analyze matrix A and return a BlockStructure.
        """
        pass
