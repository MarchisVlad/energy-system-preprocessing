from abc import ABC, abstractmethod
import scipy.sparse as sp
from ..core.BlockStructure import BlockStructure

class DetectionAlgorithm(ABC):
    """
    Abstract umbrella for all block detection algorithms.
    """

    name: str

    @abstractmethod
    def detect(self, A: sp.spmatrix) -> BlockStructure:
        """
        Analyze matrix A and return a BlockStructure.
        """
        pass
