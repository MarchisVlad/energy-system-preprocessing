from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Top-level directories
SRC    = PROJECT_ROOT / "src"
DATA   = Path("/data") / "energy-system-preprocessing"
MODELS        = DATA / "models"
PRESOLVE_RUNS = DATA / "presolve" / "runs"

# Submodule paths
SOLVERS    = SRC / "solvers"
PRESOLVERS = SRC / "presolvers"
DETECTION  = SRC / "detection"
GENERATION = SRC / "generation"

# External tool binaries (built from submodules)
PIPS_PATH      = SOLVERS    / "PIPS-IPMpp"
PAPILO_PATH    = PRESOLVERS / "papilo" / "bin" / "build" / "papilo"
PIPSTOOLS_PATH = DETECTION  / "pipstools"
SIMPLE_METHODS_PATH = GENERATION / "simple-methods"
