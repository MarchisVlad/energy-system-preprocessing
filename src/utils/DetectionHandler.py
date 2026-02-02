from src import BlockStructure
from src.core import Model

from ..detection.DetectionAlgorithm import DetectionAlgorithm
from ..detection.PatternDetection import PatternDetection
from ..detection.RCMDetection import RCMDetection
from ..detection.SpectralDetection import SpectralDetection


class DetectionHandler:

    def detect_block_structure(self,
                               model: Model,
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
            algorithm = self._choose_detection_method(model)

        if method == 'metis':
            pass
        elif method == 'spectral':
            algorithm = SpectralDetection(n_blocks=n_blocks)
        elif method == 'rcm':
            algorithm = RCMDetection()
        elif method == 'pattern':
            algorithm = RCMDetection()
        else:
            raise ValueError(f"Unknown method: {method}")

        return algorithm.detect(model)

    def _choose_detection_method(self, model: Model) -> DetectionAlgorithm:
        """Choose best detection method based on matrix properties"""
        # For small matrices, use spectral
        if model.n_rows < 1000:
            return SpectralDetection()
        # For very sparse matrices, try pattern recognition first
        elif model.A.nnz / (model.n_rows * model.n_cols) < 0.01:
            return PatternDetection()
        # Default to RCM for medium/large matrices
        else:
            return RCMDetection()
