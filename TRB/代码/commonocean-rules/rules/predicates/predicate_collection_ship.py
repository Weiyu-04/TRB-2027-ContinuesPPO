import enum
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Union
from shapely.ops import unary_union, cascaded_union
from itertools import combinations
import numpy as np
import warnings

from commonroad.geometry.shape import Shape, ShapeGroup, Polygon, Rectangle, Circle
from rules.common.ship import Ship
from rules.common.helper import OperatingMode

class PredicateCollection(ABC):
    def __init__(self, simulation_param: Dict, traffic_rules_param: Dict,
                 necessary_predicates: Set[str]):
        """
        Constructor

        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        """
        self._simulation_param = simulation_param
        self._traffic_rules_param = traffic_rules_param
        #self._country = simulation_param.get("country")
        self._necessary_predicates = necessary_predicates

    @abstractmethod
    def evaluate_predicates(self, ego_ship: Ship, other_ships: List[Ship],time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_ship: ego vehicle object containing trajectory and other relevant information
        :param other_ships: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :param operating_mode: operating mode which should be used for evaluation (monitor, constraint, or robustness)

        :returns dictionary with traces of bool values for each predicate
        """
        pass

    @abstractmethod
    def evaluate_constraints_at_time_step(self, ego_vehicle: Ship, other_vehicles: List[Ship], time_step: int) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_step: time step for which the predicates should be evaluated
        :returns dictionary with traces of bool values for each predicate
        """
        pass

    @abstractmethod
    def evaluate_constraints(self, ego_vehicle: Ship, other_vehicles: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts constraints for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with traces of constraints for each predicate
        """
        pass

    @abstractmethod
    def evaluate_robustness(self, ego_vehicle: Ship, other_vehicles: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts robustness values for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with traces of robustness values for each predicate
        """
        pass


@enum.unique
class ConstraintRepresentation(enum.Enum):
    """
    Defines the representation of a constraint
    """
    UPPER = 'upper'  # real-valued upper constraint
    LOWER = 'lower'  # real-valued lower constraint
    OUTER_BOUNDARY = 'outer_boundary'  # CommonRoad shape as an outer boundary
    INNER_BOUNDARY = 'inner_boundary'  # CommonRoad shape as an inner boundary



@enum.unique
class ConstraintType(enum.Enum):
    """
    Defines the type of constraint axis
    """
    LONGITUDINAL_CURVILINEAR_POSITION = 's'
    LATERAL_CURVILINEAR_POSITION = 'd'
    X_CARTESIAN_POSITION = 'x'
    Y_CARTESIAN_POSITION = 'y'
    VELOCITY = 'v'
    ORIENTATION = 'theta'
    ACCELERATION = 'a'
    ANGULAR_VELOCITY = 'theta_dot'
    SAFE_DISTANCE = 'd_safe'
    UNSAFE_CARTESIAN_AREA = 's_unsafe'
    UNSAFE_VELOCITY_SPACE = 'v_unsafe'


class Constraint:
    """
    Representation of a constraint so that constraints can be uses outside of the CommonOcean monitor.
    """

    def __init__(self, axis: List[ConstraintType], constraint_representation: ConstraintRepresentation,
                 value: Union[int, float, Shape, ShapeGroup, Polygon, Rectangle, Circle]):
        """
        Constructor

        :param axis: list of axis values
        :param constraint_representation: dictionary with parameters of the simulation environment
        :param value: constraint: can be a CommonRoad shape or a real value.
        """
        self._axis = axis
        self._constraint_representation = constraint_representation
        self._value = value

    @property
    def axis(self) -> List[ConstraintType]:
        return self._axis

    @property
    def constraint_representation(self) -> ConstraintRepresentation:
        return self._constraint_representation

    @property
    def value(self) -> Union[int, float, Shape, ShapeGroup, Polygon, Rectangle, Circle]:
        return self._value


class ConstraintEvaluation:
    """
    Class to extract a set of constraints from predicates
    """
    def __init__(self, predicate_collections: List[PredicateCollection]):
        self._predicate_collections = predicate_collections

    def evaluate_constraints(self, ego_ship: Ship, other_ships: List[Ship],
                             time_interval: Tuple[int, int]) -> Dict[int, List[Constraint]]:
        """
        Iterates over all predicate collections and computes constraints for a time interval

        :param ego_vehicle: ego vehicle object
        :param other_vehicles: list of other vehicles
        :param time_interval: time interval for which the predicates should be evaluated
        """
        collection_constraints = []
        constraints_per_time_step = {}
        for collection in self._predicate_collections:
            collection_constraints.append(collection.evaluate_predicates(ego_ship, other_ships, time_interval,
                                                                         OperatingMode.CONSTRAINT))
        for collection in collection_constraints:  # TODO update if only relevant predicats are returned
            for predicate_name, constraints_per_vehicle in collection.items():
                if not any(constraints_per_vehicle.values()):
                    continue
                for vehicle, time_step_constraints in constraints_per_vehicle.items():
                    if not any(time_step_constraints.values()):
                        continue
                    for time_step, constraint in time_step_constraints.items():
                        if constraints_per_time_step.get(time_step) is not None:
                            constraints_per_time_step[time_step].append(constraint)
                        else:
                            constraints_per_time_step[time_step] = [constraint]
        for time_step, constraint_list in constraints_per_time_step.items():
            constraints_per_time_step[time_step] = self.unify_constraints(constraint_list)

        return constraints_per_time_step

    @staticmethod
    def unify_constraints(constraints: List[Constraint]) -> List[Constraint]:
        """
        Combines a list of constraints for each state

        :param constraints: list of constraints
        :returns list of unified constraints
        """
        # order constraints based on their type
        ordered_constraints = {}
        unified_constraints = []
        for constr in constraints:
            constr_types = '-'.join([constr_type.value for constr_type in constr.axis]) + '-' +\
                           constr.constraint_representation.value

            if ordered_constraints.get(constr_types) is None:
                ordered_constraints[constr_types] = [constr.value]
            else:
                ordered_constraints[constr_types].append(constr.value)

        # combine constraints of same type
        for key, value in ordered_constraints.items():
            if ConstraintRepresentation.LOWER.value in key:
                unified_constraints.append(Constraint([c_type for c_type in ConstraintType if c_type.value in key],
                                                      ConstraintRepresentation.LOWER, max(value)))
            elif ConstraintRepresentation.UPPER.value in key:
                unified_constraints.append(Constraint([ctype for ctype in ConstraintType if ctype.value in key],
                                                      ConstraintRepresentation.UPPER, min(value)))
            elif ConstraintRepresentation.OUTER_BOUNDARY.value in key:
                shapely_polygons = unary_union([poly.shapely_object for poly in value])
                if shapely_polygons.geom_type == 'MultiPolygon':
                    constr_value = ShapeGroup([Polygon(np.array([[x, y] for x, y in poly.exterior.coords]))
                                               for poly in list(shapely_polygons)])
                else:
                    constr_value = Polygon(np.array([[x, y] for x, y in shapely_polygons.exterior.coords]))

                unified_constraints.append(Constraint([ctype for ctype in ConstraintType if ctype.value in key],
                                                      ConstraintRepresentation.OUTER_BOUNDARY, constr_value))
            elif ConstraintRepresentation.INNER_BOUNDARY.value in key:
                shapely_polygon = cascaded_union([a.intersection(b) for a, b in
                                                  combinations([poly.shapely_object for poly in value], 2)])
                if shapely_polygon.is_empty:
                    warnings.warn('<ConstraintEvaluation/unify_constraints>: constraint is empty set')
                    constr_value = None
                else:
                    constr_value = Polygon(np.array([[x, y] for x, y in shapely_polygon.exterior.coords][:-1]))

                unified_constraints.append(Constraint([c_type for c_type in ConstraintType if c_type.value in key],
                                           ConstraintRepresentation.INNER_BOUNDARY, constr_value))
        return unified_constraints

