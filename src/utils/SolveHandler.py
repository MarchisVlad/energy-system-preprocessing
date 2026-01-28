from src.core.Model import Model
from src.core.Solving import Solver
import subprocess
from pathlib import Path

def solve_pips(m: Model, **kwargs):
    # Extract arguments
    block_count = kwargs.get('block-count', 1)
    blocks = kwargs.get('blocks', [])
    options = kwargs.get('options', {})
    
    # Construct command components
    n_procs = block_count + 1
    model_path = Path(kwargs.get('model-path', 'SOMEFOLDER/model')).resolve()
    gams_path = Path(kwargs.get('gams-path', 'GAMSFOLDER')).resolve()
    scale_geo = options.get('scaleGeo', '')
    step_lp = options.get('stepLp', '')
    presolve = options.get('presolve', '')
    
    # Full command
    cmd = [
        'mpirun', '-np', str(n_procs),
        str(kwargs.get('pips-path', 'PIPSMAINPATH/build/gmspips')),
        str(n_procs), str(model_path), str(gams_path),
        scale_geo, step_lp, presolve
    ]
    
    # Remove empty strings in case some options are missing
    cmd = [c for c in cmd if c]
    
    # Run the solver
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Solver failed:", e.stderr)
        raise e

class SolveHandler:

    @staticmethod
    def solve(model: Model, solver: Solver = Solver.PIPS, **kwargs):
        match solver:
            case Solver.PIPS:
                return solve_pips(model, **kwargs)
            case _:
                raise NotImplementedError(f"Solver {s} not implemented")
