from abc import ABC, abstractmethod

from ..core.BlockStructure import BlockStructure
from ..core.Model import Model


class DetectionAlgorithm(ABC):
    """
    Abstract umbrella for all block detection algorithms.
    """

    name: str

    @abstractmethod
    def __init__(self, model: Model):
        self.A = model.A

    @abstractmethod
    def detect(self) -> BlockStructure:
        """
        Analyze matrix A and return a BlockStructure.
        """
        pass
