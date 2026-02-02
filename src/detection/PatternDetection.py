import scipy.sparse as sp

from ..core.BlockStructure import BlockStructure
from .DetectionAlgorithm import DetectionAlgorithm


class PatternDetection(DetectionAlgorithm):
    name = "pattern"

    def __init__(self, A: sp.coo_matrix):
        super().__init__(A)

    def detect(self, **kwargs):
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
