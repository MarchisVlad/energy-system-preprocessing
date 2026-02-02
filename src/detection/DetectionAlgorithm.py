from abc import ABC, abstractmethod
from enum import Enum

import scipy.sparse as sp

from ..core.BlockStructure import BlockStructure
from ..core.Model import Model


class DetectionAlgorithm(ABC):
    """
    Abstract umbrella for all block detection algorithms.
    """

    name: str

    @abstractmethod
    def __init__(self, A: sp.coo_matrix):
        self.A = A
        self.n_rows, self.n_cols = A.shape

    @abstractmethod
    def detect(self, **kwargs) -> BlockStructure:
        """
        Analyze matrix A and return a BlockStructure.
        """
        pass
