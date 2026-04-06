import subprocess
from pathlib import Path

from src.config import PAPILO_PATH

from ..core.model import Model
from ..core.presolving import Presolver, PresolvingMethod
from ..presolvers.techniques.coefficient_strengthening import \
    CoefficientStrengthening


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
            case Presolver.Static:

                algorithm_map = {
                    PresolvingMethod.CoeffTightening: CoefficientStrengthening
                }
                if method not in algorithm_map:
                    raise ValueError(f"Unknown algorithm: {method}")

                presolving_class = algorithm_map[method]

                # Filter kwargs to only include valid parameters for the specific reorderer
                import inspect
                sig = inspect.signature(presolving_class.__init__)
                valid_params = set(sig.parameters.keys()) - {'self'}
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k in valid_params
                }

                return presolving_class(**filtered_kwargs)
