"""Module for matrix reordering algorithms.

This module provides various graph/matrix reordering algorithms that can improve
computational efficiency of sparse matrix operations by reducing fill-in during
factorization or improving cache locality.

Notes
-----
    All reordering algorithms return permutation vectors that can be applied
    to reorder the rows and columns of a sparse matrix. The algorithms primarily
    work on the graph structure derived from the sparsity pattern of the matrix.
"""

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import metis
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank
from scipy.sparse.linalg import spilu, splu

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sksparse.cholmod import cholesky
    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False

try:
    import metis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False


class ReorderingAlgorithm(Enum):
    """Enumeration of available reordering algorithms."""

    CUTHILL_MCKEE = "CUTHILL_MCKEE"
    """Cuthill-McKee algorithm for bandwidth reduction."""

    REVERSE_CUTHILL_MCKEE = "REVERSE_CUTHILL_MCKEE"
    """Reverse Cuthill-McKee algorithm (RCM)."""

    SLOAN = "SLOAN"
    """Sloan's algorithm for profile and wavefront reduction."""

    AMD = "AMD"
    """Approximate Minimum Degree algorithm."""

    MMD = "MMD"
    """Multiple Minimum Degree algorithm."""

    NESTED_DISSECTION = "NESTED_DISSECTION"
    """Nested Dissection algorithm."""

    KING = "KING"
    """King's algorithm (variant of RCM)."""

    SPECTRAL = "SPECTRAL"
    """Spectral reordering using eigenvectors."""

    NATURAL = "NATURAL"
    """Natural ordering (identity permutation)."""

    RANDOM = "RANDOM"
    """Random permutation."""

    @classmethod
    def values(cls):
        """Return all algorithm names."""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


def reorder(A: sp.coo_matrix,
            algorithm: Union[ReorderingAlgorithm, str],
            symmetric: bool = True,
            **kwargs) -> Tuple[np.ndarray, dict]:
    """
    Apply a reordering algorithm to the matrix.
    
    Parameters
    ----------
    algorithm : ReorderingAlgorithm or str
        The reordering algorithm to apply.
    symmetric : bool, default=True
        Whether to assume the matrix is symmetric and apply symmetric
        permutation (same permutation to rows and columns).
    **kwargs : dict
        Additional parameters specific to the algorithm.
    
    Returns
    -------
    perm : np.ndarray
        Permutation vector where perm[i] = j means old row i goes to new row j.
    info : dict
        Dictionary containing information about the reordering (e.g., bandwidth,
        profile, fill-in estimates).
    """
    algorithm_map = {
        ReorderingAlgorithm.CUTHILL_MCKEE:
            CuthillMcKeeReorderer,
        ReorderingAlgorithm.REVERSE_CUTHILL_MCKEE:
            CuthillMcKeeReorderer(reverse=True),
        ReorderingAlgorithm.SLOAN:
            SloanReorderer,
        ReorderingAlgorithm.AMD:
            AMDReorderer,
        ReorderingAlgorithm.MMD:
            MMDReorderer,
        ReorderingAlgorithm.NESTED_DISSECTION:
            NestedDissectionReorderer,
        ReorderingAlgorithm.KING:
            KingReorderer,
        ReorderingAlgorithm.SPECTRAL:
            SpectralReorderer,
        ReorderingAlgorithm.NATURAL:
            NaturalReorderer,
        ReorderingAlgorithm.RANDOM:
            RandomReorderer,
    }

    if algorithm not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    reorderer_class = algorithm_map[algorithm]

    # Filter kwargs to only include valid parameters for the specific reorderer
    import inspect
    sig = inspect.signature(reorderer_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return reorderer_class(**filtered_kwargs)


class Reorderer(ABC):
    """Abstract base class for matrix reordering algorithms.
    
    All reordering algorithms should inherit from this class and implement
    the reorder() method. Subclasses can define algorithm-specific parameters
    in their __init__ methods.
    """

    def __init__(self):
        """Initialize the reorderer."""
        pass

    @abstractmethod
    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply the reordering algorithm to a sparse matrix.
        
        Parameters
        ----------
        matrix : sp.spmatrix
            Input sparse matrix to reorder.
        symmetric : bool, default=True
            Whether to assume the matrix is symmetric and apply symmetric
            permutation (same permutation to rows and columns).
        
        Returns
        -------
        perm : np.ndarray
            Permutation vector where perm[i] = j means old row i goes to new row j.
        info : dict
            Dictionary containing information about the reordering (e.g., bandwidth,
            profile, quality metrics).
        """
        pass

    @staticmethod
    def apply_permutation(matrix: sp.spmatrix,
                          perm: np.ndarray,
                          symmetric: bool = True) -> sp.spmatrix:
        """
        Apply a permutation to a sparse matrix.
        
        Parameters
        ----------
        matrix : sp.spmatrix
            Input matrix.
        perm : np.ndarray
            Permutation vector.
        symmetric : bool, default=True
            If True, apply the same permutation to both rows and columns (P * A * P^T).
            If False, only permute rows (P * A).
        
        Returns
        -------
        sp.spmatrix
            Reordered matrix.
        """
        if symmetric:
            return matrix[perm, :][:, perm]
        else:
            return matrix[perm, :]

    @staticmethod
    def _compute_bandwidth(matrix: sp.spmatrix) -> int:
        """Compute the bandwidth of a sparse matrix."""
        A_csr = matrix.tocsr()
        n = A_csr.shape[0]
        bandwidth = 0

        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            if row_end > row_start:
                cols = A_csr.indices[row_start:row_end]
                max_dist = np.max(np.abs(cols - i))
                bandwidth = max(bandwidth, max_dist)

        return int(bandwidth)

    @staticmethod
    def _compute_profile(matrix: sp.spmatrix) -> int:
        """Compute the profile (envelope size) of a sparse matrix."""
        A_csr = matrix.tocsr()
        n = A_csr.shape[0]
        profile = 0

        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            if row_end > row_start:
                cols = A_csr.indices[row_start:row_end]
                min_col = np.min(cols)
                profile += (i - min_col + 1)

        return int(profile)

    @staticmethod
    def _compute_rms_wavefront(matrix: sp.spmatrix) -> float:
        """Compute RMS wavefront size."""
        A_csr = matrix.tocsr()
        n = A_csr.shape[0]
        wavefronts = []

        for i in range(n):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            if row_end > row_start:
                cols = A_csr.indices[row_start:row_end]
                min_col = np.min(cols)
                wavefront_size = i - min_col + 1
                wavefronts.append(wavefront_size)

        if wavefronts:
            return float(np.sqrt(np.mean(np.array(wavefronts)**2)))
        return 0.0

    def _compute_quality_metrics(self, matrix: sp.spmatrix,
                                 perm: np.ndarray) -> Dict:
        """Compute quality metrics for a reordering."""
        original_bandwidth = self._compute_bandwidth(matrix)
        original_profile = self._compute_profile(matrix)
        original_wavefront = self._compute_rms_wavefront(matrix)

        A_reordered = self.apply_permutation(matrix, perm, symmetric=True)

        return {
            'original_bandwidth': original_bandwidth,
            'original_profile': original_profile,
            'original_rms_wavefront': original_wavefront,
            'bandwidth': self._compute_bandwidth(A_reordered),
            'profile': self._compute_profile(A_reordered),
            'rms_wavefront': self._compute_rms_wavefront(A_reordered),
        }


class CuthillMcKeeReorderer(Reorderer):
    """Cuthill-McKee algorithm for bandwidth reduction."""

    def __init__(self, reverse=False):
        """Initialize Cuthill-McKee reorderer."""
        super().__init__()
        self.reverse = reverse

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply Cuthill-McKee reordering."""
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        # Get RCM ordering and reverse it to get CM
        perm = reverse_cuthill_mckee(matrix, symmetric_mode=symmetric)

        if not self.reverse:
            perm = perm[::-1]

        info = {
            'algorithm': 'Cuthill-McKee',
            'symmetric': symmetric,
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info


class SloanReorderer(Reorderer):
    """Sloan's algorithm for profile and wavefront reduction."""

    def __init__(self, w1: float = 1.0, w2: float = 2.0):
        """
        Initialize Sloan reorderer.
        
        Parameters
        ----------
        w1 : float, default=1.0
            Weight for vertex degree.
        w2 : float, default=2.0
            Weight for distance from end vertex.
        """
        super().__init__()
        self.w1 = w1
        self.w2 = w2

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply Sloan reordering."""

        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        # Convert to NetworkX graph
        G = nx.from_scipy_sparse_array(matrix)

        # Find peripheral vertices
        try:
            start_node = self._find_peripheral_node(G)
            end_node = self._find_peripheral_node(G, start_node)
        except:
            start_node = 0
            end_node = len(G) - 1

        # Compute distances from end node
        distances = nx.single_source_shortest_path_length(G, end_node)

        # Compute priorities
        priorities = {}
        for node in G.nodes():
            degree = G.degree(node)
            dist = distances.get(node, 0)
            priorities[node] = self.w2 * dist - self.w1 * degree

        # Sort nodes by priority (descending)
        perm = sorted(G.nodes(), key=lambda x: priorities[x], reverse=True)
        perm = np.array(perm)

        info = {
            'algorithm': 'Sloan',
            'symmetric': symmetric,
            'w1': self.w1,
            'w2': self.w2,
            'start_node': start_node,
            'end_node': end_node,
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info

    @staticmethod
    def _find_peripheral_node(G, start: Optional[int] = None) -> int:
        """Find a peripheral node using pseudo-diameter algorithm."""
        if start is None:
            start = min(G.nodes(), key=lambda x: G.degree(x))

        current = start
        while True:
            distances = nx.single_source_shortest_path_length(G, current)
            max_dist = max(distances.values())
            farthest_nodes = [
                node for node, dist in distances.items() if dist == max_dist
            ]
            next_node = min(farthest_nodes, key=lambda x: G.degree(x))

            if next_node == current:
                break

            current = next_node

        return current


class AMDReorderer(Reorderer):
    """Approximate Minimum Degree algorithm for fill-in minimization."""

    def __init__(self):
        """Initialize AMD reorderer."""
        super().__init__()

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply AMD reordering."""
        # Try CHOLMOD if available
        if HAS_CHOLMOD and symmetric:
            try:
                A_sym = matrix + matrix.T
                A_sym = A_sym.tocsc()
                factor = cholesky(A_sym, ordering_method='amd')
                perm = factor.P()

                info = {
                    'algorithm': 'AMD (CHOLMOD)',
                    'symmetric': symmetric,
                }
                info.update(self._compute_quality_metrics(matrix, perm))

                return perm, info
            except Exception as e:
                warnings.warn(f"CHOLMOD AMD failed: {e}. Using fallback.",
                              UserWarning)

        # Fallback implementation
        perm = self._amd_fallback(matrix)

        info = {
            'algorithm': 'AMD (fallback)',
            'symmetric': symmetric,
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info

    @staticmethod
    def _amd_fallback(matrix: sp.spmatrix) -> np.ndarray:
        """Fallback AMD implementation using minimum degree heuristic."""
        n = matrix.shape[0]
        A = matrix.tocsr()

        eliminated = np.zeros(n, dtype=bool)
        perm = []

        # Build adjacency structure
        adjacency = [set() for _ in range(n)]
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            for j in range(row_start, row_end):
                col = A.indices[j]
                if col != i:
                    adjacency[i].add(col)
                    adjacency[col].add(i)

        # Eliminate vertices in order of minimum degree
        for _ in range(n):
            min_degree = float('inf')
            min_vertex = -1

            for v in range(n):
                if not eliminated[v]:
                    degree = sum(1 for u in adjacency[v] if not eliminated[u])
                    if degree < min_degree:
                        min_degree = degree
                        min_vertex = v

            if min_vertex == -1:
                break

            eliminated[min_vertex] = True
            perm.append(min_vertex)

            # Update adjacency
            neighbors = [u for u in adjacency[min_vertex] if not eliminated[u]]
            for u in neighbors:
                for v in neighbors:
                    if u != v:
                        adjacency[u].add(v)
                        adjacency[v].add(u)

        return np.array(perm)


class MMDReorderer(Reorderer):
    """Multiple Minimum Degree algorithm."""

    def __init__(self):
        """Initialize MMD reorderer."""
        super().__init__()

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply MMD reordering (currently aliased to AMD)."""
        warnings.warn("MMD is not fully implemented. Using AMD instead.",
                      UserWarning)
        amd = AMDReorderer()
        perm, info = amd.reorder(matrix, symmetric=symmetric)
        info['algorithm'] = 'MMD (AMD fallback)'
        return perm, info


class NestedDissectionReorderer(Reorderer):
    """Nested Dissection algorithm using recursive graph partitioning."""

    def __init__(self, max_depth: int = 5):
        """
        Initialize Nested Dissection reorderer.
        
        Parameters
        ----------
        max_depth : int, default=5
            Maximum recursion depth for spectral bisection fallback.
        """
        super().__init__()
        self.max_depth = max_depth

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply Nested Dissection reordering."""
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        if HAS_METIS:
            try:
                perm = self._metis_nested_dissection(matrix)
                info = {
                    'algorithm': 'Nested Dissection (METIS)',
                    'symmetric': symmetric,
                }
                info.update(self._compute_quality_metrics(matrix, perm))
                return perm, info
            except Exception as e:
                warnings.warn(
                    f"METIS nested dissection failed: {e}. Using fallback.",
                    UserWarning)

        # Fallback to spectral bisection
        perm = self._spectral_nested_dissection(matrix)

        info = {
            'algorithm': 'Nested Dissection (spectral)',
            'symmetric': symmetric,
            'max_depth': self.max_depth,
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info

    @staticmethod
    def _metis_nested_dissection(matrix: sp.spmatrix) -> np.ndarray:
        """Use METIS for nested dissection ordering."""
        n = matrix.shape[0]
        A = matrix.tocsr()

        # Build adjacency lists
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            for j in range(row_start, row_end):
                col = A.indices[j]
                if col != i and col not in adjacency[i]:
                    adjacency[i].append(col)

        perm, iperm = metis.nested_dissection(adjacency)
        return np.array(perm)

    def _spectral_nested_dissection(self, matrix: sp.spmatrix) -> np.ndarray:
        """Nested dissection using spectral bisection."""
        n = matrix.shape[0]

        def recursive_bisection(indices, depth=0):
            if len(indices) <= 10 or depth >= self.max_depth:
                return indices

            sub_matrix = matrix[indices, :][:, indices]
            separator_indices = self._find_spectral_separator(sub_matrix)

            if len(separator_indices) == 0 or len(separator_indices) == len(
                    indices):
                return indices

            all_indices_set = set(range(len(indices)))
            sep_set = set(separator_indices)
            remaining = all_indices_set - sep_set

            if len(remaining) == 0:
                return indices

            remaining_list = list(remaining)
            mid = len(remaining_list) // 2
            part1_local = remaining_list[:mid]
            part2_local = remaining_list[mid:]

            part1 = [indices[i] for i in part1_local]
            part2 = [indices[i] for i in part2_local]
            separator = [indices[i] for i in separator_indices]

            ordered_part1 = recursive_bisection(part1, depth + 1)
            ordered_part2 = recursive_bisection(part2, depth + 1)

            return ordered_part1 + ordered_part2 + separator

        perm = recursive_bisection(list(range(n)))
        return np.array(perm)

    @staticmethod
    def _find_spectral_separator(matrix: sp.spmatrix) -> List[int]:
        """Find a separator using spectral partitioning."""
        if matrix.shape[0] <= 3:
            return []

        D = sp.diags(np.array(matrix.sum(axis=1)).flatten())
        L = D - matrix

        try:
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(L.tocsc(),
                                              k=2,
                                              which='SM',
                                              maxiter=1000)
            fiedler = eigenvectors[:, 1]
        except:
            n = matrix.shape[0]
            separator_size = max(1, n // 10)
            return list(np.random.choice(n, separator_size, replace=False))

        median = np.median(fiedler)
        std = np.std(fiedler)
        separator_threshold = 0.5 * std
        separator = np.where(np.abs(fiedler - median) < separator_threshold)[0]

        return separator.tolist()


class KingReorderer(Reorderer):
    """King's algorithm - variant of RCM with improved peripheral vertex selection."""

    def __init__(self):
        """Initialize King reorderer."""
        super().__init__()

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply King reordering."""
        if not HAS_NETWORKX:
            warnings.warn("NetworkX not available. Falling back to RCM.",
                          UserWarning)
            rcm = ReverseCuthillMcKeeReorderer()
            return rcm.reorder(matrix, symmetric=symmetric)

        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        G = nx.from_scipy_sparse_array(matrix)
        start = self._find_peripheral_node(G)

        # BFS from peripheral vertex
        visited = []
        queue = [start]
        visited_set = {start}

        while queue:
            queue.sort(key=lambda x: G.degree(x))
            node = queue.pop(0)
            visited.append(node)

            neighbors = [n for n in G.neighbors(node) if n not in visited_set]
            for neighbor in neighbors:
                if neighbor not in visited_set:
                    queue.append(neighbor)
                    visited_set.add(neighbor)

        perm = np.array(visited[::-1])

        info = {
            'algorithm': 'King',
            'symmetric': symmetric,
            'start_node': start,
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info

    @staticmethod
    def _find_peripheral_node(G, start: Optional[int] = None) -> int:
        """Find a peripheral node using pseudo-diameter algorithm."""
        if start is None:
            start = min(G.nodes(), key=lambda x: G.degree(x))

        current = start
        while True:
            distances = nx.single_source_shortest_path_length(G, current)
            max_dist = max(distances.values())
            farthest_nodes = [
                node for node, dist in distances.items() if dist == max_dist
            ]
            next_node = min(farthest_nodes, key=lambda x: G.degree(x))

            if next_node == current:
                break

            current = next_node

        return current


class SpectralReorderer(Reorderer):
    """Spectral reordering using Fiedler vector."""

    def __init__(self):
        """Initialize Spectral reorderer."""
        super().__init__()

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply Spectral reordering."""
        n = matrix.shape[0]

        if symmetric:
            A = matrix + matrix.T
        else:
            A = matrix

        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - A

        try:
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(L.tocsc(),
                                              k=2,
                                              which='SM',
                                              maxiter=1000)
            fiedler = eigenvectors[:, 1]
        except Exception as e:
            warnings.warn(
                f"Spectral ordering failed: {e}. Using natural ordering.",
                UserWarning)
            perm = np.arange(n)
            info = {
                'algorithm': 'Spectral (failed)',
                'error': str(e),
            }
            return perm, info

        perm = np.argsort(fiedler)

        info = {
            'algorithm': 'Spectral',
            'symmetric': symmetric,
            'eigenvalues': eigenvalues.tolist(),
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info


class NaturalReorderer(Reorderer):
    """Natural ordering (identity permutation)."""

    def __init__(self):
        """Initialize Natural reorderer."""
        super().__init__()

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply Natural ordering (no reordering)."""
        n = matrix.shape[0]
        perm = np.arange(n)

        info = {
            'algorithm': 'Natural',
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info


class RandomReorderer(Reorderer):
    """Random permutation."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Random reorderer.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__()
        self.seed = seed

    def reorder(self,
                matrix: sp.spmatrix,
                symmetric: bool = True) -> Tuple[np.ndarray, Dict]:
        """Apply Random reordering."""
        if self.seed is not None:
            np.random.seed(self.seed)

        n = matrix.shape[0]
        perm = np.random.permutation(n)

        info = {
            'algorithm': 'Random',
            'seed': self.seed,
        }
        info.update(self._compute_quality_metrics(matrix, perm))

        return perm, info
