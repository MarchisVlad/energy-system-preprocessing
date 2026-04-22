from __future__ import annotations

import subprocess
from pathlib import Path

from src.core.model import Model
from src.core.presolving import PresolvingMethod

_BINARY = Path(__file__).parent / "build" / "papilo_handler"


def presolve(model: Model, method: PresolvingMethod) -> Model:
    """
    Apply a single PaPILO presolve technique in one round on one thread.

    The input model must have an MPS file path (model.path).  The reduced
    problem is written next to the original file with a ``_reduced`` suffix
    and returned as a new Model loaded from that path.
    """
    if not _BINARY.exists():
        raise RuntimeError(
            f"PaPILO handler binary not found at {_BINARY}.\n"
            "Build it with:\n"
            "  cd src/presolvers/papilo_handler && mkdir -p build && "
            "cd build && cmake .. && make"
        )

    if not model.path or not model.path.endswith(".mps"):
        raise ValueError(
            "PaPILO handler requires the model to have an MPS file path "
            "(model.path must end with .mps)."
        )

    input_path = Path(model.path)
    output_path = input_path.with_stem(input_path.stem + "_reduced")

    result = subprocess.run(
        [str(_BINARY), str(input_path), str(output_path), method.name],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"PaPILO handler failed for method '{method.name}':\n{result.stderr}"
        )

    return Model(path=str(output_path))
