from ..core.BlockStructure import BlockStructure
from .DetectionAlgorithm import DetectionAlgorithm


class PatternDetection(DetectionAlgorithm):
    name = "pattern"

    def __init__(self, model):
        super().__init__(model)

    def detect(self):
        row_perm, col_perm, blocks = self._run_rcm(self.A)

        return BlockStructure(blocks=blocks,
                              count=len(blocks),
                              row_permutation=row_perm,
                              col_permutation=col_perm,
                              method=self.name,
                              detected_patterns=["banded"],
                              metadata={
                                  "heuristic": "reverse_cuthill_mckee",
                                  "block_extraction": "sliding_window"
                              })
