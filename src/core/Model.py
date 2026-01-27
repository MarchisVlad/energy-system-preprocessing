import gamspy as gp 
from enum import Enum
import src.core 
import src.detection

class ModelType(Enum):
    MPS = 1
    GMS = 2
    GDX = 3
    LP  = 4

class Model(gp.Model):

    blocks: BlockStructure
    presolves: Optional[List[PresolvingMethod]] = None # e.g. "CoeffTightening"

    pass
    

