from enum import Enum


class Presolver(Enum):
    PaPILO = 1
    Static = 2


class PresolvingMethod(Enum):
    CoeffTightening = 1
    Propagation = 2
    ColSingleton = 3
    DualFix = 4
    FixContinuous = 5
    ParallelCols = 6
    ParallelRows = 7
    SimpleProbing = 8
    DoubleToNEq = 9
    SimpifyIneq = 10
    Stuffing = 11
    DomCol = 12
    DualInfer = 13
    ImplInt = 14
    Probing = 15
    Sparsify = 16

class PresolveStatus(Enum):
    kUnchanged = "unchanged"
    kReduced = "reduced"
    kInfeasible = "infeasible"
