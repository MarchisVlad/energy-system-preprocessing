import random
import tempfile
import warnings

import gamspy as gp
import mip
import numpy as np
import pytest
import scipy.sparse as sp

from src.config import MODELS
from src.core.model import FileFormat, Model, Problem, Sense


def test_sense_enum():
    assert Sense.MIN.value == "MIN"
    assert str(Sense.MAX) == "MAX"
    assert set(Sense.values()) == {"MIN", "MAX", "FEASIBILITY"}


def test_problem_enum():
    assert Problem.LP.value == "LP"
    assert str(Problem.MIP) == "MIP"
    assert set(Problem.values()) == {"LP", "MIP", "NLP", "QCP"}


def test_file_format_enum():
    assert FileFormat.CPLEXLP.value == "cplex.lp"
    assert str(FileFormat.CPLEXMPS) == "cplex.mps"
    assert set(FileFormat.values()) == {
        "cplex.lp", "cplex.mps", "dict.txt", "dictmap.gdx", "jacobian.gdx",
        "execute_unload.gdx"
    }


def test_model_mps_load_from_file():

    model_path = list((MODELS / "formats").rglob("*.mps"))

    assert model_path, "No MPS files found in models/formats/"

    mps_model = random.choice(model_path)
    model = Model(path=str(mps_model), format=FileFormat.CPLEXMPS)

    assert model.format == FileFormat.CPLEXMPS
    assert model.A is not None
    assert isinstance(model.model, mip.Model)


def test_model_gdx_load_from_file():

    model_path = list(
        (MODELS / "formats" / "gdx" / "simple_models").rglob("*.gdx"))

    assert model_path, "No GDX files found in models/formats/"

    gdx_model = random.choice(model_path)
    model = Model(path=str(gdx_model), format=FileFormat.GDXUnload)

    assert model.format == FileFormat.GDXUnload
    assert model.A is not None
    assert isinstance(model.model, gp.Model)


def test_model_constrain_matrix_properties():

    ten_teams_path = MODELS / "formats" / "mps" / "10teams" / "10teams.mps"
    ten_teams_model = Model(path=str(ten_teams_path),
                            format=FileFormat.CPLEXMPS)

    assert sp.issparse(ten_teams_model.A), "A should be a sparse matrix"
    assert ten_teams_model.A.shape == (230, 2025)
    assert ten_teams_model.A.nnz == 12150
