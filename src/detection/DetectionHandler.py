from . import RCMDetection, SpectralDetection, PatternDetection
from src.core.Model import Model


class DetectionHandler:

    def detect_block_structure(model: Model,
                               method='auto',
                               n_blocks=None,
                               min_block_size=10,
                               threshold=0.1):
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
            method = self._choose_detection_method(model)

        if method == 'metis':
            return self._detect_metis(n_blocks)
        elif method == 'spectral':
            return self._detect_spectral(n_blocks)
        elif method == 'rcm':
            return self._detect_rcm()
        elif method == 'pattern':
            return self._detect_patterns(threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _choose_detection_method(self, model: Model):
        """Choose best detection method based on matrix properties"""
        # For small matrices, use spectral
        if model.n_rows < 1000:
            return 'spectral'
        # For very sparse matrices, try pattern recognition first
        elif model.A.nnz / (model.n_rows * model.n_cols) < 0.01:
            return 'pattern'
        # Default to RCM for medium/large matrices
        else:
            return 'rcm'
