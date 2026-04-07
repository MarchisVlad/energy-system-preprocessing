import random
import tempfile
import warnings

import mip
import numpy as np
import pytest
import scipy.sparse as sp

from src.config import MODELS
from src.core.model import FileFormat, Model


def test_model_mps_load_from_file():

    model_path = list((MODELS / "formats").rglob("*.mps"))

    assert model_path, "No MPS files found in models/formats/"

    mps_model = random.choice(model_path)

    print(f"Testing model loading from file: {mps_model}")

    model = Model(path=str(mps_model), format=FileFormat.CPLEXMPS)

    assert model.format == FileFormat.CPLEXMPS
    assert model.A is not None
    assert isinstance(model.model, mip.Model)


def test_model_constrain_matrix_properties():

    ten_teams_path = MODELS / "formats" / "mps" / "10teams" / "10teams.mps"
    ten_teams_model = Model(path=str(ten_teams_path), format=FileFormat.CPLEXMPS)

    assert sp.issparse(ten_teams_model.A), "A should be a sparse matrix"
    assert ten_teams_model.A.shape == (230, 2025)
    assert ten_teams_model.A.nnz == 12150
