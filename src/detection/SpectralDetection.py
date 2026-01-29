from .DetectionAlgorithm import DetectionAlgorithm
from ..core.BlockStructure import BlockStructure
from sklearn import SpectralClustering

class SpectralDetection(DetectionAlgorithm):
    name = "spectral"

    def __init__(self, n_blocks=None):
        self.n_blocks = n_blocks

    def detect(self, A):
        similarity = (A @ A.T).astype(float).toarray()

        clustering = SpectralClustering(
            n_clusters=self.n_blocks,
            affinity="precomputed",
            random_state=42
        )

        labels = clustering.fit_predict(similarity)
        blocks = self._blocks_from_labels(A, labels)

        return BlockStructure(
            blocks=blocks,
            count=len(blocks)
        )
