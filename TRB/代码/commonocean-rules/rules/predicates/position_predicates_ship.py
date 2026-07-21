from typing import Set
from rules.common.utils_ship import *

from rules.predicates.predicate_collection_ship import PredicateCollection
from rules.common.ship import Ship
from rules.common.helper import OperatingMode


class PositionPredicateCollection(PredicateCollection):
    def __init__(self, simulation_param: Dict, traffic_rules_param: Dict, necessary_predicates: Set[str]):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        :param traffic_sign_interpreter: CommonOcean traffic sign interpreter
        """
        super().__init__(simulation_param, traffic_rules_param, necessary_predicates)

    @staticmethod
    def in_front_sector(time_step: int, ship_p: Ship, ship_k: Ship, head_on_angle: float) -> bool:
        """
        Evaluates if the kth ship is in front sector (+- 5 degree to the heading) of the pth ship

        :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        degree = head_on_angle
        orientation_up = ship_p.states_cr[time_step].orientation + (degree*2*np.pi/360)
        orientation_low = ship_p.states_cr[time_step].orientation - (degree*2*np.pi/360)
        if (in_halfspace(ship_k.states_cr[time_step].position,ship_p.states_cr[time_step].position,orientation_up,True) and
                in_halfspace(ship_k.states_cr[time_step].position,ship_p.states_cr[time_step].position,orientation_low,False)):
            return True
        else:
            return False

    @staticmethod
    def in_left_sector(time_step: int, ship_p: Ship, ship_k: Ship, head_on_angle: float, overtake_angle: float) -> bool:
        """
        Evaluates if the kth ship is in left sector (202.5 - 355 degree to heading) of the pth ship

        :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        orientation_up = ship_p.states_cr[time_step].orientation + (head_on_angle*2*np.pi/360)
        orientation_low = ship_p.states_cr[time_step].orientation + (overtake_angle*4*np.pi/360)
        if (in_halfspace(ship_k.states_cr[time_step].position,ship_p.states_cr[time_step].position,orientation_up,False) and
                in_halfspace(ship_k.states_cr[time_step].position,ship_p.states_cr[time_step].position,orientation_low,True)):
            return True
        else:
            return False

    @staticmethod
    def in_right_sector(time_step: int, ship_p: Ship, ship_k: Ship, head_on_angle: float, overtake_angle: float) -> bool:
        """
        Evaluates if the kth ship is in right sector (5 - 157.5 degree to heading) of the pth ship

       :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        orientation_up = ship_p.states_cr[time_step].orientation - (head_on_angle*2*np.pi/360)
        orientation_low = ship_p.states_cr[time_step].orientation - (overtake_angle*4*np.pi/360)
        if (in_halfspace(ship_k.states_cr[time_step].position,ship_p.states_cr[time_step].position,orientation_up,True) and
                in_halfspace(ship_k.states_cr[time_step].position,ship_p.states_cr[time_step].position,orientation_low,False)):
            return True
        else:
            return False

    @staticmethod
    def in_behind_sector(time_step: int, ship_p: Ship, ship_k: Ship, overtake_angle:float) -> bool:
        """
        Evaluates if the pth ship is in behind sector (between 112.5 and -112.5 degree to the heading) of the kth ship

        :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        degree = overtake_angle*2
        orientation_up = ship_k.states_cr[time_step].orientation + (degree*2*np.pi/360)
        orientation_low = ship_k.states_cr[time_step].orientation - (degree*2*np.pi/360)
        if (in_halfspace(ship_p.states_cr[time_step].position,ship_k.states_cr[time_step].position,orientation_up,False) and
                in_halfspace(ship_p.states_cr[time_step].position,ship_k.states_cr[time_step].position,orientation_low,True)):
            return True
        else:
            return False

    def _ships_left_sector(self, ship: Ship, other_ships: List[Ship], time_step: int) -> List[Ship]:
        """
        Searches for ships left of a ship

        :param ship: ship object
        :param other_ships: other ships in scenario
        :param time_step: time step of interest
        :returns list of ships left of a ship
        """
        left_ships = []
        for other_ship in other_ships:
            if self.in_left_sector(time_step,ship,other_ship,
                                   self._traffic_rules_param.get("max_orientation_diff_head_on"),
                                   self._traffic_rules_param.get("max_orientation_diff_overtake")):
                left_ships.append(other_ship)
        return left_ships

    def _ships_right_sector(self, ship: Ship, other_ships: List[Ship],  time_step: int) -> List[Ship]:
        """
        Searches for ships right of a ship

        :param ship: ship object
        :param other_ships: other ships in scenario
        :param time_step: time step of interest
        :returns list of ships right of a ship
        """
        right_ships = []
        for other_ship in other_ships:
            if self.in_right_sector(time_step,ship,other_ship,
                                    self._traffic_rules_param.get("max_orientation_diff_head_on"),
                                    self._traffic_rules_param.get("max_orientation_diff_overtake")):
                right_ships.append(other_ship)
        return right_ships

    def _ships_front_sector(self, ship: Ship, other_ships: List[Ship],  time_step: int) -> List[Ship]:
        """
        Searches for ships front of a ship

        :param ship: ship object
        :param other_ships: other ships in scenario
        :param time_step: time step of interest
        :returns list of ships front of a ship
        """
        front_ships = []
        for other_ship in other_ships:
            if self.in_front_sector(time_step, ship, other_ship,
                                    self._traffic_rules_param.get("max_orientation_diff_head_on")):
                front_ships.append(other_ship)
        return front_ships

    @staticmethod
    def orientation_delta(time_step: int, ship_p: Ship, ship_k: Ship, max_orientation_diff: float, offset: float = None) -> bool:
        """
        Evaluates if the heading of the kth ship differs more than 10 degree from the pth ship's heading

        :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :param max_orientation_diff: maximum allowable orientation difference to be seen as no difference
        :param offset: if orientation delta with respect to offset is necessary
        :returns boolean indicating satisfaction
        """
        # try:
        if offset:
            diff = wrap_angle((ship_p.states_cr[time_step].orientation + offset) - ship_k.states_cr[time_step].orientation)
        else:
            diff = wrap_angle(ship_p.states_cr[time_step].orientation - ship_k.states_cr[time_step].orientation)
        # except ValueError:
        #     print(time_step)
        if diff < ((max_orientation_diff/360)*2*np.pi) or diff > (((360 - max_orientation_diff)/360) * 2 * np.pi):
            return False
        else:
            return True

    @staticmethod
    def orientation_towards_left(time_step: int, ship_p: Ship, ship_k: Ship, head_on_angle: float) -> bool:
        """
        Evaluates if the heading of the kth ship is orientated left from the perspective of pth ship

        :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        # try:
        min_angle = wrap_angle(ship_p.states_cr[time_step].orientation + ((head_on_angle/360)*2*np.pi))
        max_angle = wrap_angle(ship_p.states_cr[time_step].orientation + (((180-head_on_angle)/360)*2*np.pi))
        if min_angle > max_angle:
            if min_angle <= wrap_angle(ship_k.states_cr[time_step].orientation) <= 2* np.pi or \
                    0.0 <= wrap_angle(ship_k.states_cr[time_step].orientation) <= max_angle:
                return True
            else:
                return False
        else:
            if min_angle <= wrap_angle(ship_k.states_cr[time_step].orientation) <= max_angle:
                return True
            else:
                return False

    @staticmethod
    def orientation_towards_right(time_step: int, ship_p: Ship, ship_k: Ship, head_on_angle: float) -> bool:
        """
        Evaluates if the heading of the kth ship is orientated right from the perspective of pth ship

        :param ship_p: pth ship
        :param ship_k: kth ship
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        # try:
        min_angle = wrap_angle(ship_p.states_cr[time_step].orientation - (((180-head_on_angle)/360)*2*np.pi))
        max_angle = wrap_angle(ship_p.states_cr[time_step].orientation - ((head_on_angle/360)*2*np.pi))
        if min_angle > max_angle:
            if min_angle <= wrap_angle(ship_k.states_cr[time_step].orientation) <= 2* np.pi or \
                    0.0 <= wrap_angle(ship_k.states_cr[time_step].orientation) <= max_angle:
                return True
            else:
                return False
        else:
            if min_angle <= wrap_angle(ship_k.states_cr[time_step].orientation) <= max_angle:
                return True
            else:
                return False

    def evaluate_predicates(self, ego_ship: Ship, other_ships: List[Ship], time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_ship: ego vehicle object containing trajectory and other relevant information
        :param other_ships: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :param operating_mode: operating mode which should be used for evaluation (monitor, constraint, or robustness)

        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"in_left_sector__x_ego_x_o": {}}

        for other_ship in other_ships:
            predicate_trace["in_left_sector__x_ego_x_o"][other_ship.id] = {}

            for time_step in ego_ship.time_steps:
                if time_step not in other_ship.time_steps:
                    continue
                if "in_left_sector__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["in_left_sector__x_ego_x_o"][other_ship.id][time_step] = \
                        self.in_left_sector(time_step, ego_ship, other_ship,
                                            self._traffic_rules_param.get("max_orientation_diff_head_on"),
                                            self._traffic_rules_param.get("max_orientation_diff_overtake"))
        return predicate_trace

    def evaluate_constraints(self, ego_ship: Ship, other_ships: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        return {}

    def evaluate_robustness(self, ego_ship: Ship, other_ships: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        pass

    def evaluate_constraints_at_time_step(self, ego_vehicle: Ship, other_vehicles: List[Ship], time_step: int) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_step: time step for which the predicates should be evaluated
        :returns dictionary with traces of bool values for each predicate
        """
        return {}