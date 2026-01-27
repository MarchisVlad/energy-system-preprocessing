from enum import Enum

class Solver(Enum):
    CPLEX = 1,
    SCIP  = 2,
    GAMS  = 3,
    PIPS  = 4