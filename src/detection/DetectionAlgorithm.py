from abc import ABC, abstractmethod
from ..core.BlockStructure import BlockStructure
from src.core.Model import Model


class DetectionAlgorithm(ABC):
    """
    Abstract umbrella for all block detection algorithms.
    """

    name: str

    @abstractmethod
    def detect(self, model: Model) -> BlockStructure:
        """
        Analyze matrix A and return a BlockStructure.
        """
        pass
