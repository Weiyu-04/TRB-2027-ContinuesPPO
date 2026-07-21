from typing import List, Dict, Set, Tuple

from rules.predicates.predicate_collection_ship import PredicateCollection, Constraint, ConstraintRepresentation, ConstraintType
from rules.common.ship import Ship
from rules.common.helper import OperatingMode


class VelocityPredicateCollection(PredicateCollection):
    def __init__(self, simulation_param: Dict, traffic_rules_param: Dict, necessary_predicates: Set[str]):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        """
        super().__init__(simulation_param, traffic_rules_param, necessary_predicates)

    @staticmethod
    def drives_faster(time_step: int, ship_k: Ship, ship_p: Ship) -> bool:
        """
        Predicate which checks if the kth ship drives faster than the pth ship

        :param ship_p: the pth ship
        :param ship_k: the kth ship
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if ship_k.states_cr[time_step].velocity <= ship_p.states_cr[time_step].velocity:
            return False
        else:
            return True

    @staticmethod
    def safe_speed(time_step: int, ship: Ship, operating_mode: OperatingMode) -> bool or List[Constraint]:
        """
        Predicate which checks if the ship drives at a safe speed (less than v_max and more than v_min)

        :param ship: the ship of interest
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if ship.vehicle_param.get("v_min") <= ship.states_cr[time_step].velocity <= ship.vehicle_param.get("v_max"):
            if operating_mode is OperatingMode.MONITOR:
                return True
            elif operating_mode is OperatingMode.CONSTRAINT:
                return []  # nothing to do as rule is not applied
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            if operating_mode is OperatingMode.MONITOR:
                return False
            elif operating_mode is OperatingMode.CONSTRAINT:
                return [Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER,
                                   ship.vehicle_param.get("v_max")),
                        Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER,
                                   ship.vehicle_param.get("v_min"))]
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def evaluate_predicates(self, ego_ship: Ship, other_ships: List[Ship], time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :param operating_mode: operating mode which should be used for evaluation (monitor, constraint, or robustness)
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"safe_speed__x_ego": {ego_ship.id: {}},
                           "large_orientation_change__x_ego": {ego_ship.id: {}},
                           "large_orientation_change_to_starboard__x_ego": {ego_ship.id: {}},
                           "no_orientation_change__x_ego": {ego_ship.id: {}},
                           "no_turn__x_ego": {ego_ship.id: {}}
                           }
        for time_step in ego_ship.time_steps:
            if "safe_speed__x_ego" in self._necessary_predicates:
                predicate_trace["safe_speed__x_ego"][ego_ship.id][time_step] = \
                    self.safe_speed(time_step, ego_ship, operating_mode)
            if "large_orientation_change__x_ego" in self._necessary_predicates:
                predicate_trace["large_orientation_change__x_ego"][ego_ship.id][time_step] = \
                    self.large_orientation_change(time_step, ego_ship)
            if "large_orientation_change_to_starboard__x_ego" in self._necessary_predicates:
                predicate_trace["large_orientation_change_to_starboard__x_ego"][ego_ship.id][time_step] = \
                    self.large_orientation_change_to_starboard(time_step, ego_ship)
            if "no_orientation_change__x_ego" in self._necessary_predicates:
                predicate_trace["no_orientation_change__x_ego"][ego_ship.id][time_step] = \
                    self.no_orientation_change(time_step, ego_ship)
            if "no_turn__x_ego" in self._necessary_predicates:
                predicate_trace["no_turn__x_ego"][ego_ship.id][time_step] = \
                    self.no_turn(time_step, ego_ship)

        return predicate_trace

    def evaluate_constraints(self, ego_ship: Ship, other_ships: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        constraint_trace = {"safe_speed__x_ego": {}}

        # constraint_trace[time_step] = copy.deepcopy(self._constraint_dict)
        for time_step in ego_ship.time_steps:
            if "safe_speed__x_ego" in self._necessary_predicates:
                constraint_trace["safe_speed__x_ego"][time_step] = self.safe_speed(time_step, ego_ship,
                                                                                   OperatingMode.CONSTRAINT)
                constraint_trace["all_constraints_at_time_step"][time_step].extend(
                    constraint_trace["safe_speed__x_ego"][time_step])
        return constraint_trace

    def evaluate_robustness(self, ego_ship: Ship, other_ships: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        pass

    def evaluate_constraints_at_time_step(self, ego_ship: Ship, other_ships: List[Ship], time_step: int) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_step: time step for which the predicates should be evaluated
        :returns dictionary with traces of bool values for each predicate
        """
        constraint_trace = {"safe_speed__x_ego": {},
                            "all_constraints_at_time_step": []}

        # constraint_trace[time_step] = copy.deepcopy(self._constraint_dict)
        if "safe_speed__x_ego" in self._necessary_predicates:
            constraint_trace["safe_speed__x_ego"] = self.safe_speed(time_step, ego_ship, OperatingMode.CONSTRAINT)
            constraint_trace["all_constraints_at_time_step"].extend(
                constraint_trace["safe_speed__x_ego"])
        return constraint_trace