from enum import Enum
from typing import Optional

from configs.plot.colours import *


class ColourEnum(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, _id: int, colour: Optional[tuple]):
        self._value_ = _id
        self.colour = colour


class NodePositions(ColourEnum):
    IS_NOT_NODE = (0, None)
    IS_NODE = (1, black)


class NodeDegrees(ColourEnum):
    IS_NOT_NODE = (0, None)
    DEG1 = (1, yellow_1)
    DEG2 = (2, green_2)
    DEG3 = (3, blue_1)
    DEG4 = (4, orange_1)
    DEG5 = (5, red_2)


class NodeTypes(ColourEnum):
    IS_NOT_NODE = (0, None)
    CROSSING = (1, green_1)
    END = (2, gold)
    BORDER = (3, red_3)



