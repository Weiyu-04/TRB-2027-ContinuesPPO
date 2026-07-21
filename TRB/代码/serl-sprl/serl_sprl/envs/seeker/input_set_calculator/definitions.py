from enum import Enum, auto
from typing import Any, Callable, Union

from continuoussets import Interval, Zonotope
from continuoussets.convexsets.convexset import ConvexSet
from gymnasium.core import ObsType

SafeInputSetComputationFn_T = Callable[[ObsType, dict[str, Any]], ConvexSet]


class ConvexSet(Enum):
    Zonotope = 1
    Polytope = 2
    Interval = 3


Set = Union[ConvexSet, Zonotope, Interval]


class ZonoOptimizationMode(int, Enum):
    """
    VOL_MAX maximizes the geometric mean of the scaling factors.
    SUPPORT_MAX maximizes the geometric mean of support functions in random directions.
    BOX_MAX maximizes the volume of an enclosed box.
    BOUND_MIN minimizes the squared distance of random points within the reach set to boundary points on XS.
    SCALE maximizes a scalar scaling factor of the template input set.
    """

    VOL_MAX = auto()
    SUPPORT_MAX = auto()
    BOX_MAX = auto()
    BOUND_MIN = auto()
    SCALE = auto()
