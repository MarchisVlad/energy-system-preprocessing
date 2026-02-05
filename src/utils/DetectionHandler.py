import scipy.sparse as sp

from ..core.BlockStructure import BlockStructure
from ..detection.DetectionAlgorithm import DetectionAlgorithm
from ..detection.PatternDetection import PatternDetection
from ..detection.RCMDetection import RCMDetection
from ..detection.SpectralDetection import SpectralDetection


class DetectionHandler:

    def detect_block_structure(self,
                               A: sp.coo_matrix,
                               method: str | None = 'auto',
                               n_blocks: int | None = 10,
                               min_block_size: int | None = 10,
                               threshold: float | None = 0.1) -> BlockStructure:
        """
        Detect block structure in constraint matrix
        
        Args:
            method: 'metis', 'spectral', 'rcm', 'pattern', or 'auto'
            n_blocks: Number of blocks (if known)
            min_block_size: Minimum rows/cols per block
            threshold: Density threshold for considering blocks separate
        
        Returns:
            BlockStructure object with partition info and reordering
        """
        if method == 'auto':
            algorithm = self._choose_detection_method(A)

        if method == 'metis':
            pass
        elif method == 'spectral':
            algorithm = SpectralDetection(A)
        elif method == 'rcm':
            algorithm = RCMDetection(A)
        elif method == 'pattern':
            algorithm = RCMDetection(A)
        else:
            raise ValueError(f"Unknown method: {method}")

        return algorithm.detect(n_blocks=None,
                                min_block_size=min_block_size,
                                threshold=threshold)

    def _choose_detection_method(self, A: sp.coo_matrix) -> DetectionAlgorithm:

        n_rows, n_cols = A.shape
        """Choose best detection method based on matrix properties"""
        # For small matrices, use spectral
        if n_rows < 1000:
            return SpectralDetection(A)
        # For very sparse matrices, try pattern recognition first
        elif A.nnz / (n_rows * n_cols) < 0.01:
            return PatternDetection(A)
        # Default to RCM for medium/large matrices
        else:
            return RCMDetection(A)
