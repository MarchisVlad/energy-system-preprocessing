from enum import Enum
from src.core.BlockStructure import BlockStructure
from src.core.Presolving import PresolvingMethod

from pathlib import Path

class ModelType(Enum):
    UNSUPPORTED = 0
    MPS = 1
    GMS = 2
    GDX = 3
    LP  = 4

class Model():

    # Relative paths containing Model information
    model_path: Path
    reduced_path: Path
    postsolve_path: Path
    solution_path: Path = None # Optional

    # Model configuration
    format: ModelType
    blocks: BlockStructure
    presolves: list(PresolvingMethod) = None # e.g. "CoeffTightening"
    original_solution: int

    def __init__(self, model_path: Path = None) -> None:
        self.model_path = model_path
        self.format = self.get_type()

    def get_type(self):
        
        model_extensions = {
            '.mps' : ModelType.MPS,
            '.gdx' : ModelType.GDX,
            '.gms' : ModelType.GMS,
            '.lp'  : ModelType.LP
        }

        extension = '.' + self.model_path.split('.')[-1].lower() if '.' in  self.model_path else ''

        return model_extensions.get(extension, ModelType.UNSUPPORTED)
    

