"""Hash-based sparse matrix cache.

If the same effective matrix appears from two different presolving orderings,
the cache avoids recomputing it. Cache key = hash(sorted list of technique slugs
applied to the same original MPS).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _cache_key(original_mps: Path, techniques: list[str]) -> str:
    """Stable hash of (absolute mps path, sorted techniques)."""
    payload = json.dumps(
        {"mps": str(original_mps.resolve()), "techniques": sorted(techniques)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class MatrixCache:
    """File-based cache mapping (mps, techniques) → scipy sparse matrix."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def get(self, original_mps: Path, techniques: list[str]) -> sp.spmatrix | None:
        """Return cached matrix or None if not cached."""
        key = _cache_key(original_mps, techniques)
        path = self._path(key)
        if not path.exists():
            return None
        data = np.load(str(path), allow_pickle=False)
        return sp.csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"]),
        )

    def put(self, original_mps: Path, techniques: list[str], matrix: sp.spmatrix) -> Path:
        """Store *matrix* under the cache key and return the file path."""
        key = _cache_key(original_mps, techniques)
        path = self._path(key)
        csr = matrix.tocsr()
        np.savez_compressed(
            str(path),
            data=csr.data,
            indices=csr.indices,
            indptr=csr.indptr,
            shape=np.array(csr.shape),
        )
        return path

    def has(self, original_mps: Path, techniques: list[str]) -> bool:
        key = _cache_key(original_mps, techniques)
        return self._path(key).exists()

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
