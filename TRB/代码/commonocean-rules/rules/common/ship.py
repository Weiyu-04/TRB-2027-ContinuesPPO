from typing import Union, Set, Dict, List
import enum

from commonroad.geometry.shape import Shape, Rectangle
from commonocean.scenario.state import State
from commonocean.scenario.obstacle import ObstacleType


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """
    __slots__ = ['s', 'v', 'a', 'j']

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def attributes(self) -> List[str]:
        """ Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = '\n'
        for attr in self.attributes:
            state += attr
            state += '= {}\n'.format(self.__getattribute__(attr))
        return state


class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """
    __slots__ = ['d', 'theta', 'kappa', 'kappa_dot']

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def attributes(self) -> List[str]:
        """ Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = '\n'
        for attr in self.attributes:
            state += attr
            state += '= {}\n'.format(self.__getattribute__(attr))
        return state


class Input:
    """
    Lateral and longitudinal vehicle input
    """
    __slots__ = ['a', 'kappa_dot_dot']

    @property
    def attributes(self) -> List[str]:
        """ Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = '\n'
        for attr in self.attributes:
            state += attr
            state += '= {}\n'.format(self.__getattribute__(attr))
        return state


@enum.unique
class VehicleClassification(enum.Enum):
    EGO_SHIP = 0
    CROSSING_SHIP = 1
    OVERTAKING_SHIP = 2
    ENCOUNTER_SHIP = 3
    OTHER_SHIP = 4


class Ship:
    """
    Representation of a ship with state and input profiles and other information for complete simulation horizon
    """
    def __init__(self, shape: Union[Shape, Rectangle], cr_state: Dict[int, State], vehicle_id: int, obstacle_type: ObstacleType,
                 vehicle_param: Dict, vehicle_classification: Dict[int, VehicleClassification] = None):
        """
        :param shape: CommonRoad shape of vehicle
        :param cr_state: initial CommonOcean state of vehicle
        :param vehicle_id: id of vehicle
        :param obstacle_type: type of the vehicle, e.g. parked car, car, bus, ...
        :param vehicle_classification: which relation is the vehicle , e.g., ego, overtaking, encounter...
        """
        self._states_cr = cr_state
        self._shape = shape
        self._id = vehicle_id
        self._obstacle_type = obstacle_type
        self._vehicle_classification = vehicle_classification
        self._vehicle_param = vehicle_param

    @property
    def shape(self) -> Rectangle:
        return self._shape

    @property
    def id(self) -> int:
        return self._id

    @property
    def time_steps(self) -> List[int]:
        time_steps = []
        for state in self._states_cr.values():
            time_steps.append(state.time_step)
        return time_steps

    @property
    def states_cr(self) -> Dict[int, State]:
        return self._states_cr

    @property
    def state_list_cr(self) -> List[State]:
        state_list = []
        for state in self._states_cr.values():
            state_list.append(state)
        return state_list

    @property
    def obstacle_type(self) -> ObstacleType:
        return self._obstacle_type

    @property
    def vehicle_param(self) -> Dict:
        return self._vehicle_param

    def append_time_step(self, time_step: int, state_cr: State):
        """
        Adds information for a specific time step to vehicle

        :param time_step: time step of new data
        :param state_cr: CommonOcean state to append
        """
        self._states_cr[time_step] = state_cr

