import inspect
from pathlib import Path

from ..core.model import Model
from ..core.presolving import Presolver, PresolvingMethod
from ..presolvers.papilo import PaPILO
from ..presolvers.static.coefficient_strengthening import CoefficientStrengthening
from ..presolvers.static.constraint_propagation import ConstraintPropagation
from ..presolvers.static.dual_fix import DualFix
from ..presolvers.static.fix_continuous import FixContinuous
from ..presolvers.static.sparsify import Sparsify


_STATIC_ALGORITHM_MAP = {
    PresolvingMethod.CoeffTightening: CoefficientStrengthening,
    PresolvingMethod.Propagation: ConstraintPropagation,
    PresolvingMethod.DualFix: DualFix,
    PresolvingMethod.FixContinuous: FixContinuous,
    PresolvingMethod.Sparsify: Sparsify,
}


class PresolveHandler:

    @staticmethod
    def presolve(
        model: Model,
        presolver: Presolver = Presolver.Static,
        method: PresolvingMethod = PresolvingMethod.CoeffTightening,
        temp_dir: Path | None = None,
        step: int = 0,
        **kwargs,
    ) -> Model:
        match presolver:
            case Presolver.PaPILO:
                if temp_dir is None:
                    raise ValueError("temp_dir is required for PaPILO")
                if not model.path:
                    raise ValueError("PaPILO requires a model with a source MPS path")

                input_mps = Path(model.path)
                output_mps = temp_dir / f"step_{step:04d}_reduced.mps"

                PaPILO(methods=[method]).apply(input_mps, output_mps)

                return Model(path=str(output_mps))

            case Presolver.Static:
                if method not in _STATIC_ALGORITHM_MAP:
                    raise ValueError(f"No static implementation for: {method}")

                presolving_class = _STATIC_ALGORITHM_MAP[method]

                sig = inspect.signature(presolving_class.__init__)
                valid_params = set(sig.parameters.keys()) - {'self'}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

                algorithm = presolving_class(**filtered_kwargs)
                algorithm._run(model)
                model.update_matrix()
                return model
