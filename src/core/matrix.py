"""Module for manipulating and interacting with sparse matrix representations.

Notes
-----
    This module provides a unified interface for working with different sparse
    matrix formats from scipy.sparse, with utilities for conversion, analysis,
    and manipulation.
"""

import warnings
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import norm, spsolve


class MatrixFormat(Enum):
    """An enumeration for sparse matrix format types"""

    COO = "COO"
    """Coordinate format (COOrdinate)."""

    CSR = "CSR"
    """Compressed Sparse Row format."""

    CSC = "CSC"
    """Compressed Sparse Column format."""

    LIL = "LIL"
    """List of Lists format."""

    DOK = "DOK"
    """Dictionary of Keys format."""

    DIA = "DIA"
    """Diagonal storage format."""

    BSR = "BSR"
    """Block Sparse Row format."""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class MatrixProperty(Enum):
    """An enumeration for matrix property types"""

    SYMMETRIC = "SYMMETRIC"
    """Matrix is symmetric."""

    POSITIVE_DEFINITE = "POSITIVE_DEFINITE"
    """Matrix is positive definite."""

    DIAGONAL = "DIAGONAL"
    """Matrix is diagonal."""

    LOWER_TRIANGULAR = "LOWER_TRIANGULAR"
    """Matrix is lower triangular."""

    UPPER_TRIANGULAR = "UPPER_TRIANGULAR"
    """Matrix is upper triangular."""

    BANDED = "BANDED"
    """Matrix is banded."""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Matrix:
    """Class representing a sparse matrix with comprehensive manipulation capabilities.
    
    Attributes
    ----------
    data : Union[sp.spmatrix, np.ndarray]
        The underlying matrix data in sparse or dense format.
    format : MatrixFormat
        The current sparse matrix format.
    shape : Tuple[int, int]
        The dimensions of the matrix (rows, cols).
    dtype : np.dtype
        The data type of matrix elements.
    nnz : int
        Number of non-zero elements.
    density : float
        Proportion of non-zero elements.
    properties : List[MatrixProperty]
        Detected structural properties of the matrix.
    metadata : dict
        Additional metadata and annotations.
    """

    def __init__(
        self,
        data: Union[sp.spmatrix, np.ndarray, List, Tuple] | None = None,
        shape: Tuple[int, int] | None = None,
        format: MatrixFormat | None = MatrixFormat.COO,
        dtype: np.dtype | None = None,
        path: str | None = None,
        properties: List[MatrixProperty] | None = None,
        metadata: dict | None = None,
    ):
        """
        Construct a Matrix instance.

        Parameters
        ----------
        data : sp.spmatrix, np.ndarray, list, or tuple, optional
            Matrix data in various formats.
        shape : Tuple[int, int], optional
            Shape of the matrix if creating empty matrix.
        format : MatrixFormat, optional
            Desired sparse matrix format. Default is CSR.
        dtype : np.dtype, optional
            Data type for matrix elements. Default is float64.
        path : str, optional
            Path to load matrix from file (supports .npz, .mtx, .mat).
        properties : List[MatrixProperty], optional
            Known structural properties of the matrix.
        metadata : dict, optional
            Additional metadata and annotations.
        """
        self.metadata = metadata if metadata is not None else {}
        self.properties = properties if properties is not None else []
        self._format = format
        self._dtype = dtype if dtype is not None else np.float64

        # Initialize from file if path specified
        if path is not None:
            self._load_from_file(path)
        elif data is not None:
            self._initialize_from_data(data)
        elif shape is not None:
            # Create empty sparse matrix
            self.data = self._create_sparse(format, shape, dtype=self._dtype)
        else:
            raise ValueError(
                "Must provide either data, shape, or path to initialize Matrix")

        self._analyze_properties()

    def _initialize_from_data(self, data: Union[sp.spmatrix, np.ndarray, List,
                                                Tuple]):
        """Initialize matrix from various data formats."""
        if isinstance(data, sp.spmatrix):
            self.data = data
            self._format = self._get_format_from_matrix(data)
        elif isinstance(data, np.ndarray):
            self.data = self._to_sparse(data, self._format)
        elif isinstance(data, (list, tuple)):
            # Assume COO format: (data, (row, col))
            if len(data) == 2 and isinstance(data[1], tuple):
                values, (rows, cols) = data
                if 'shape' in self.metadata:
                    shape = self.metadata['shape']
                else:
                    shape = (max(rows) + 1, max(cols) + 1)
                self.data = sp.coo_matrix((values, (rows, cols)),
                                          shape=shape,
                                          dtype=self._dtype)
                self._format = MatrixFormat.COO
            else:
                # Treat as dense array
                self.data = self._to_sparse(np.array(data), self._format)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _create_sparse(self, format: MatrixFormat, shape: Tuple[int, int],
                       dtype: np.dtype) -> sp.spmatrix:
        """Create an empty sparse matrix of specified format."""
        format_map = {
            MatrixFormat.COO: sp.coo_matrix,
            MatrixFormat.CSR: sp.csr_matrix,
            MatrixFormat.CSC: sp.csc_matrix,
            MatrixFormat.LIL: sp.lil_matrix,
            MatrixFormat.DOK: sp.dok_matrix,
            MatrixFormat.DIA: sp.dia_matrix,
            MatrixFormat.BSR: sp.bsr_matrix,
        }
        return format_map[format](shape, dtype=dtype)

    def _get_format_from_matrix(self, matrix: sp.spmatrix) -> MatrixFormat:
        """Determine the MatrixFormat from a scipy sparse matrix."""
        format_map = {
            sp.coo_matrix: MatrixFormat.COO,
            sp.csr_matrix: MatrixFormat.CSR,
            sp.csc_matrix: MatrixFormat.CSC,
            sp.lil_matrix: MatrixFormat.LIL,
            sp.dok_matrix: MatrixFormat.DOK,
            sp.dia_matrix: MatrixFormat.DIA,
            sp.bsr_matrix: MatrixFormat.BSR,
        }
        for matrix_type, format_enum in format_map.items():
            if isinstance(matrix, matrix_type):
                return format_enum
        return MatrixFormat.CSR  # Default fallback

    def _to_sparse(self, array: np.ndarray,
                   format: MatrixFormat) -> sp.spmatrix:
        """Convert dense array to sparse format."""
        format_map = {
            MatrixFormat.COO: sp.coo_matrix,
            MatrixFormat.CSR: sp.csr_matrix,
            MatrixFormat.CSC: sp.csc_matrix,
            MatrixFormat.LIL: sp.lil_matrix,
            MatrixFormat.DOK: sp.dok_matrix,
            MatrixFormat.DIA: sp.dia_matrix,
            MatrixFormat.BSR: sp.bsr_matrix,
        }
        return format_map[format](array, dtype=self._dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the matrix."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of matrix elements."""
        return self.data.dtype

    @property
    def nnz(self) -> int:
        """Get the number of non-zero elements."""
        return self.data.nnz

    @property
    def density(self) -> float:
        """Calculate the density (proportion of non-zero elements)."""
        return self.nnz / (self.shape[0] * self.shape[1])

    @property
    def format(self) -> MatrixFormat:
        """Get the current matrix format."""
        return self._format

    def convert(self, format: MatrixFormat) -> 'Matrix':
        """
        Convert matrix to a different sparse format.

        Parameters
        ----------
        format : MatrixFormat
            Target sparse matrix format.

        Returns
        -------
        Matrix
            A new Matrix object in the specified format.
        """
        if self._format == format:
            warnings.warn(f"Matrix is already in format {format}", UserWarning)
            return self

        conversion_map = {
            MatrixFormat.COO: lambda m: m.tocoo(),
            MatrixFormat.CSR: lambda m: m.tocsr(),
            MatrixFormat.CSC: lambda m: m.tocsc(),
            MatrixFormat.LIL: lambda m: m.tolil(),
            MatrixFormat.DOK: lambda m: m.todok(),
            MatrixFormat.DIA: lambda m: m.todia(),
            MatrixFormat.BSR: lambda m: m.tobsr(),
        }

        new_data = conversion_map[format](self.data)
        return Matrix(
            data=new_data,
            format=format,
            properties=self.properties.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dense(self) -> np.ndarray:
        """
        Convert sparse matrix to dense numpy array.

        Returns
        -------
        np.ndarray
            Dense representation of the matrix.
        """
        return self.data.toarray()

    def copy(self) -> 'Matrix':
        """
        Create a deep copy of the matrix.

        Returns
        -------
        Matrix
            A new Matrix object with copied data.
        """
        return Matrix(
            data=self.data.copy(),
            format=self._format,
            properties=self.properties.copy(),
            metadata=self.metadata.copy(),
        )

    def transpose(self) -> 'Matrix':
        """
        Transpose the matrix.

        Returns
        -------
        Matrix
            Transposed matrix.
        """
        return Matrix(
            data=self.data.transpose(),
            format=self._format,
            properties=self.properties.copy(),
            metadata=self.metadata.copy(),
        )

    def get_row(self, index: int) -> 'Matrix':
        """
        Extract a single row.

        Parameters
        ----------
        index : int
            Row index.

        Returns
        -------
        Matrix
            Matrix containing only the specified row.
        """
        row_data = self.data[index, :]
        return Matrix(data=row_data, format=self._format)

    def get_col(self, index: int) -> 'Matrix':
        """
        Extract a single column.

        Parameters
        ----------
        index : int
            Column index.

        Returns
        -------
        Matrix
            Matrix containing only the specified column.
        """
        col_data = self.data[:, index]
        return Matrix(data=col_data, format=self._format)

    def get_submatrix(self, rows: Union[slice, List[int]],
                      cols: Union[slice, List[int]]) -> 'Matrix':
        """
        Extract a submatrix.

        Parameters
        ----------
        rows : slice or List[int]
            Row indices or slice.
        cols : slice or List[int]
            Column indices or slice.

        Returns
        -------
        Matrix
            Extracted submatrix.
        """
        submatrix_data = self.data[rows, cols]
        return Matrix(data=submatrix_data, format=self._format)

    def eliminate_zeros(self) -> 'Matrix':
        """
        Remove zero entries from the matrix.

        Returns
        -------
        Matrix
            Matrix with zeros eliminated.
        """
        new_data = self.data.copy()
        new_data.eliminate_zeros()
        return Matrix(
            data=new_data,
            format=self._format,
            properties=self.properties.copy(),
            metadata=self.metadata.copy(),
        )

    def _analyze_properties(self):
        """Analyze and detect structural properties of the matrix."""
        if self.shape[0] != self.shape[1]:
            # Non-square matrix
            return

        # Check if diagonal
        if self._is_diagonal():
            if MatrixProperty.DIAGONAL not in self.properties:
                self.properties.append(MatrixProperty.DIAGONAL)

        # Check if symmetric
        if self._is_symmetric():
            if MatrixProperty.SYMMETRIC not in self.properties:
                self.properties.append(MatrixProperty.SYMMETRIC)

        # Check if triangular
        if self._is_lower_triangular():
            if MatrixProperty.LOWER_TRIANGULAR not in self.properties:
                self.properties.append(MatrixProperty.LOWER_TRIANGULAR)

        if self._is_upper_triangular():
            if MatrixProperty.UPPER_TRIANGULAR not in self.properties:
                self.properties.append(MatrixProperty.UPPER_TRIANGULAR)

    def _is_diagonal(self) -> bool:
        """Check if matrix is diagonal."""
        if self.shape[0] != self.shape[1]:
            return False
        csr = self.data.tocsr()
        for i in range(self.shape[0]):
            row_start = csr.indptr[i]
            row_end = csr.indptr[i + 1]
            for j in range(row_start, row_end):
                col = csr.indices[j]
                if col != i and abs(csr.data[j]) > 1e-10:
                    return False
        return True

    def _is_symmetric(self, tol: float = 1e-10) -> bool:
        """Check if matrix is symmetric."""
        if self.shape[0] != self.shape[1]:
            return False
        diff = self.data - self.data.transpose()
        return np.allclose(diff.data, 0, atol=tol)

    def _is_lower_triangular(self, tol: float = 1e-10) -> bool:
        """Check if matrix is lower triangular."""
        if self.shape[0] != self.shape[1]:
            return False
        csr = self.data.tocsr()
        for i in range(self.shape[0]):
            row_start = csr.indptr[i]
            row_end = csr.indptr[i + 1]
            for j in range(row_start, row_end):
                col = csr.indices[j]
                if col > i and abs(csr.data[j]) > tol:
                    return False
        return True

    def _is_upper_triangular(self, tol: float = 1e-10) -> bool:
        """Check if matrix is upper triangular."""
        if self.shape[0] != self.shape[1]:
            return False
        csr = self.data.tocsr()
        for i in range(self.shape[0]):
            row_start = csr.indptr[i]
            row_end = csr.indptr[i + 1]
            for j in range(row_start, row_end):
                col = csr.indices[j]
                if col < i and abs(csr.data[j]) > tol:
                    return False
        return True

    def norm(self, ord: Union[str, int, float] = 'fro') -> float:
        """
        Compute matrix norm.

        Parameters
        ----------
        ord : str, int, or float
            Order of the norm. Common values: 'fro' (Frobenius), 1, 2, np.inf.

        Returns
        -------
        float
            The computed norm.
        """
        return norm(self.data, ord=ord)

    def _load_from_file(self, path: str):
        """
        Load matrix from file.

        Supports .npz (scipy sparse), .mtx (Matrix Market), and .mat (MATLAB).
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        if suffix == '.npz':
            loaded = sp.load_npz(path)
            self.data = loaded
            self._format = self._get_format_from_matrix(loaded)
        elif suffix == '.mtx':
            loaded = sio.mmread(path)
            if not sp.issparse(loaded):
                loaded = sp.csr_matrix(loaded)
            self.data = loaded
            self._format = self._get_format_from_matrix(loaded)
        elif suffix == '.mat':
            mat_dict = sio.loadmat(path)
            # Assume the matrix is stored under a key (needs specification)
            key = self.metadata.get('mat_key', 'matrix')
            if key in mat_dict:
                loaded = mat_dict[key]
                if not sp.issparse(loaded):
                    loaded = sp.csr_matrix(loaded)
                self.data = loaded
                self._format = self._get_format_from_matrix(loaded)
            else:
                raise KeyError(f"Key '{key}' not found in .mat file")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def save(self, path: str, format: str | None = None):
        """
        Save matrix to file.

        Parameters
        ----------
        path : str
            Path to save the matrix.
        format : str, optional
            File format ('npz', 'mtx', 'mat'). If None, inferred from extension.
        """
        path_obj = Path(path)
        if format is None:
            format = path_obj.suffix.lower()[1:]  # Remove leading dot

        if format == 'npz':
            sp.save_npz(path, self.data)
        elif format == 'mtx':
            sio.mmwrite(path, self.data)
        elif format == 'mat':
            key = self.metadata.get('mat_key', 'matrix')
            sio.savemat(path, {key: self.data})
        else:
            raise ValueError(f"Unsupported save format: {format}")

    def __repr__(self) -> str:
        """String representation of the Matrix."""
        return (f"Matrix(shape={self.shape}, format={self.format}, "
                f"nnz={self.nnz}, density={self.density:.4f}, "
                f"dtype={self.dtype})")

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()

    def __add__(self, other: Union['Matrix', sp.spmatrix,
                                   np.ndarray]) -> 'Matrix':
        """Add two matrices."""
        if isinstance(other, Matrix):
            result = self.data + other.data
        else:
            result = self.data + other
        return Matrix(data=result, format=self._format)

    def __sub__(self, other: Union['Matrix', sp.spmatrix,
                                   np.ndarray]) -> 'Matrix':
        """Subtract two matrices."""
        if isinstance(other, Matrix):
            result = self.data - other.data
        else:
            result = self.data - other
        return Matrix(data=result, format=self._format)

    def __mul__(
            self, other: Union['Matrix', sp.spmatrix, np.ndarray,
                               float]) -> 'Matrix':
        """Element-wise multiplication or scalar multiplication."""
        if isinstance(other, Matrix):
            result = self.data.multiply(other.data)
        elif isinstance(other, (int, float)):
            result = self.data * other
        else:
            result = self.data.multiply(other)
        return Matrix(data=result, format=self._format)

    def __matmul__(self, other: Union['Matrix', sp.spmatrix,
                                      np.ndarray]) -> 'Matrix':
        """Matrix multiplication."""
        if isinstance(other, Matrix):
            result = self.data @ other.data
        else:
            result = self.data @ other
        return Matrix(data=result, format=self._format)

    def __getitem__(self, key):
        """Index into the matrix."""
        result = self.data[key]
        if sp.issparse(result):
            return Matrix(data=result, format=self._format)
        return result

    def __setitem__(self, key, value):
        """Set values in the matrix."""
        self.data[key] = value


class MatrixHistory:
    """Wrapper for tracking matrix transformations and operations.

    Attributes
    ----------
    states : List[Matrix]
        The history of matrix states.
    operations : List[str]
        Description of operations applied at each step.
    current_index : int
        The current index in the history.
    logs : List[str]
        Detailed logging information.
    """

    def __init__(self, matrix: Matrix):
        """
        Initialize a new matrix history.

        Parameters
        ----------
        matrix : Matrix
            The initial matrix state.
        """
        self.states: List[Matrix] = [matrix.copy()]
        self.operations: List[str] = ["Initial state"]
        self.current_index: int = 0
        self.logs: List[str] = []

    def add_state(self, matrix: Matrix, operation: str, log: str | None = None):
        """
        Add a new state to the history.

        Parameters
        ----------
        matrix : Matrix
            The new matrix state.
        operation : str
            Description of the operation performed.
        log : str, optional
            Additional logging information.
        """
        # Truncate history if we're not at the end
        self.states = self.states[:self.current_index + 1]
        self.operations = self.operations[:self.current_index + 1]

        self.states.append(matrix.copy())
        self.operations.append(operation)
        self.current_index = len(self.states) - 1

        if log:
            self.logs.append(f"[{operation}] {log}")

    def get_current_state(self) -> Matrix | None:
        """Get the current matrix state."""
        if 0 <= self.current_index < len(self.states):
            return self.states[self.current_index]
        return None

    def revert_to_index(self, index: int) -> bool:
        """
        Revert to a specific state in history.

        Parameters
        ----------
        index : int
            The history index to revert to.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if 0 <= index < len(self.states):
            self.current_index = index
            return True
        return False

    def get_state_at_index(self, index: int) -> Matrix | None:
        """Get the matrix state at a specific index."""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None

    def get_history_summary(self) -> List[str]:
        """Get a summary of all operations in the history."""
        return self.operations.copy()

    def undo(self) -> bool:
        """Undo the last operation."""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def redo(self) -> bool:
        """Redo the next operation."""
        if self.current_index < len(self.states) - 1:
            self.current_index += 1
            return True
        return False

    def clear_history(self):
        """Clear all history except the current state."""
        current = self.get_current_state()
        if current:
            self.states = [current.copy()]
            self.operations = ["Current state"]
            self.current_index = 0
            self.logs = []

    def __repr__(self) -> str:
        """String representation of the history."""
        return (f"MatrixHistory(states={len(self.states)}, "
                f"current_index={self.current_index})")
