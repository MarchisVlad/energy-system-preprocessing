import subprocess
from pathlib import Path

from src.config import PAPILO_PATH

from ..core.Model import Model
from ..core.Presolving import Presolver, PresolvingMethod


def presolve_papilo(model: Model,
                    method: PresolvingMethod = PresolvingMethod.CoeffTightening,
                    **kwargs) -> Model:
    """
    Call the PaPILO presolver on the model and optionally postsolve it.
    """

    # Presolve options (currently only method name, can expand)
    method_flag = kwargs.get('method-flag', method.name.lower())

    # Build the presolve command
    cmd_presolve = [
        str(PAPILO_PATH), 'presolve', '-f',
        str(model.model_path.absolute()), '-r',
        str(model.reduced_path.absolute()), '-o',
        str(model.postsolve_path.absolute())
    ]

    try:
        print(f"Running PaPILO presolve: {' '.join(cmd_presolve)}")
        result = subprocess.run(cmd_presolve,
                                check=True,
                                capture_output=True,
                                text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("PaPILO presolve failed:", e.stderr)
        raise e

    return model


class PresolveHandler:

    def presolve(model: Model,
                 presolver: Presolver = Presolver.Static,
                 method: PresolvingMethod = PresolvingMethod.CoeffTightening,
                 **kwargs) -> Model:
        match presolver:
            case Presolver.PaPILO:
                return presolve_papilo(model, method, **kwargs)
            case _:
                raise NotImplementedError(
                    f"Presolver {presolver} not implemented")
