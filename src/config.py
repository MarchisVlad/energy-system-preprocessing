from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Main file structure 
SRC         = PROJECT_ROOT / 'src'
DATA        = PROJECT_ROOT / 'data'
EXPERIMENTS = PROJECT_ROOT / 'experiments'
MODELS      = PROJECT_ROOT / 'models'

# Solver paths
SOLVERS = SRC / 'solvers'
PIPS_PATH = SOLVERS / 'PIPS-IPMpp'

# Presolver paths
PRESOLVERS = SRC / 'presolvers'
PAPILO_PATH = PRESOLVERS / 'papilo' / 'bin' / 'build' / 'papilo'
