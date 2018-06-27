from enum import Enum

class ActivationType(Enum):
    NONE = 0 # can use this for linear case
    SIGMOID = 1
    RELU = 2

class LossType(Enum):
    SQUARED_ERROR = 0
    CROSS_ENTROPY = 1

class LayerType(Enum):
    INPUT = 0
    FULLY_CONNECTED = 1
    OUTPUT = 4

class RegType(Enum):
    L1 = 0
    L2 = 1

class GradType(Enum):
    WEIGHT = 0
    BIAS = 1

class TargetType(Enum):
    NOVEC = 0
    CPU = 1
