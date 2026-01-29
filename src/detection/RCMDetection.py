from .DetectionAlgorithm import DetectionAlgorithm
from ..core.BlockStructure import BlockStructure

class RCMDetection(DetectionAlgorithm):
    name = "rcm"

    def detect(self, A):
        row_perm, col_perm, blocks = self._run_rcm(A)

        return BlockStructure(
            blocks=blocks,
            count=len(blocks),
            row_permutation=row_perm,
            col_permutation=col_perm,
            method=self.name,
            detected_patterns=["banded"],
            metadata={
                "heuristic": "reverse_cuthill_mckee",
                "block_extraction": "sliding_window"
            }
        )
