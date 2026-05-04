"""Per-model folder management.

Each model lives under DATA_ROOT/models/{name}/ with the layout:
    original.mps
    metadata.json
    matrix.npz                         (optional)
    detection_original.json
    presolved_{method}_{backend}.mps
    detection_{method}_{backend}.json
    presolve_debug_{method}_{backend}.txt
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import MODELS


class ModelStore:
    """Manage the artifact folder for a single model."""

    def __init__(self, name: str, root: Path | None = None):
        self.name = name
        self.root = (root or MODELS) / name
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def original_mps(self) -> Path:
        return self.root / "original.mps"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def matrix_cache_path(self) -> Path:
        return self.root / "matrix.npz"

    def presolved_mps(self, slug: str) -> Path:
        return self.root / f"presolved_{slug}.mps"

    def detection_json(self, stage: str) -> Path:
        return self.root / f"detection_{stage}.json"

    def debug_txt(self, slug: str) -> Path:
        return self.root / f"presolve_debug_{slug}.txt"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def load_metadata(self) -> dict:
        if self.metadata_path.exists():
            return json.loads(self.metadata_path.read_text())
        return {}

    def save_metadata(self, extra: dict | None = None) -> None:
        meta = self.load_metadata()
        if extra:
            meta.update(extra)
        meta["name"] = self.name
        meta["updated_at"] = datetime.utcnow().isoformat()
        self.metadata_path.write_text(json.dumps(meta, indent=2))

    def init_from_mps(self, mps_path: Path) -> None:
        """Copy *mps_path* into this model's folder as original.mps and
        populate metadata.json with basic MPS stats."""
        import shutil

        from src.core.mps_reader import read_metadata

        shutil.copy2(mps_path, self.original_mps)

        stats = read_metadata(self.original_mps)
        self.save_metadata(
            {
                "source": str(mps_path),
                "created_at": datetime.utcnow().isoformat(),
                "n_vars": stats.n_vars,
                "n_constraints": stats.n_constraints,
                "nnz": stats.nnz,
                "n_integer_vars": stats.n_integer_vars,
                "has_objective": stats.has_objective,
            }
        )

    # ------------------------------------------------------------------
    # Detection results
    # ------------------------------------------------------------------

    def save_detection(self, stage: str, result_dict: dict) -> Path:
        """Write a DetectionResult.to_dict() to disk and return the path."""
        path = self.detection_json(stage)
        path.write_text(json.dumps(result_dict, indent=2))
        return path

    def load_detection(self, stage: str) -> dict | None:
        path = self.detection_json(stage)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_stages(self) -> list[str]:
        """Return slugs for which presolved MPS files exist."""
        return [
            p.stem.removeprefix("presolved_")
            for p in self.root.glob("presolved_*.mps")
        ]

    def list_detections(self) -> list[str]:
        """Return stage names for which detection JSON files exist."""
        return [
            p.stem.removeprefix("detection_")
            for p in self.root.glob("detection_*.json")
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def list_all(cls, root: Path | None = None) -> list[str]:
        """Return the names of all models in the store root."""
        store_root = root or MODELS
        if not store_root.exists():
            return []
        return [d.name for d in store_root.iterdir() if d.is_dir()]
