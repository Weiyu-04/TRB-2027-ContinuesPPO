from typing import Set, Union

from rules.predicates.predicate_collection_ship import PredicateCollection
from rules.predicates.velocity_predicates_ship import VelocityPredicateCollection
from rules.predicates.position_predicates_ship import PositionPredicateCollection

from rules.common.utils_ship import *

from rules.common.helper import OperatingMode


class GeneralPredicateCollection(PredicateCollection):
    def __init__(self, simulation_param: Dict, traffic_rules_param: Dict, necessary_predicates: Set[str], constraint_param: Dict):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        :param constraint_param: dictionary with parameters for constraining the utilizable state space
        """
        super().__init__(simulation_param, traffic_rules_param, necessary_predicates)
        self._constraint_dict = constraint_param
        self.time_step_before_keep = -1
        self.current_delta_orientation_keep = 0
        self.time_step_before_crossing = -1
        self.current_delta_orientation_crossing = 0
        self.time_step_before_overtake = -1
        self.current_delta_orientation_overtake = 0
        self.time_step_before_head_on = -1
        self.current_delta_orientation_head_on = 0

    @staticmethod
    def safe_distance(time_step: int, ship: Ship, other_ship: Ship, t_react: float, operating_mode: OperatingMode) -> Union[bool, Constraint, List[Constraint]]:
        """
        Old predicate version - not used anymore

        Evaluates if the safe distance is kept between two ships, whereby the safe distance is defined through the distance
        both ships would sail until the stop when full braking and the reaction time for the ego ship

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship: other ship
        :param t_react: Reaction time of ego ship [s]
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :return: Boolean indicating satisfaction
        """
        # the distance between to ships is safe, if both could stop before
        d_safe = calculate_safe_distance(ship, other_ship, time_step, t_react)
        if d_safe < np.linalg.norm(ship.states_cr[time_step].position - other_ship.states_cr[time_step].position):
            if operating_mode is OperatingMode.MONITOR:
                return True
            elif operating_mode is OperatingMode.CONSTRAINT:
                return [Constraint([ConstraintType.SAFE_DISTANCE], ConstraintRepresentation.LOWER, d_safe)]
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            if operating_mode is OperatingMode.MONITOR:
                return False
            elif operating_mode is OperatingMode.CONSTRAINT:
                return [Constraint([ConstraintType.SAFE_DISTANCE], ConstraintRepresentation.LOWER, d_safe),
                        Constraint([ConstraintType.ACCELERATION], ConstraintRepresentation.UPPER, 0)]
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def collision_possible(self,time_step: int, ship: Ship, other_ship: Ship):

        d_AB = np.linalg.norm(ship.states_cr[time_step].position - other_ship.states_cr[time_step].position)
        if d_AB < np.absolute(ship.shape.length + other_ship.shape.length):
            return True
        else:
            VO = construct_velocity_obstacle(ship.states_cr[time_step], other_ship.states_cr[time_step],
                                             ship.shape.length, other_ship.shape.length)

            v_other = velocity_vector(other_ship.states_cr[time_step])
            v_ship = velocity_vector(ship.states_cr[time_step])
            v_AB = relative_velocity_vector(ship.states_cr[time_step],other_ship.states_cr[time_step])
            VO.translate_rotate(v_other,0.0)
            tip_velocity = v_ship + ship.states_cr[time_step].position

            if VO.contains_point(tip_velocity) and np.any(tip_velocity != ship.states_cr[time_step].position) and \
                    np.linalg.norm(v_AB) >= (d_AB / self._traffic_rules_param.get('t_horizon_collision_possible')):
                return True
            else:
                return False

    def collision_possible_soon(self,time_step: int, ship: Ship, other_ship: Ship, operating_mode: OperatingMode) -> Union[bool, Constraint, List[Constraint]]:

        d_AB = np.linalg.norm(ship.states_cr[time_step].position - other_ship.states_cr[time_step].position)
        if d_AB < np.absolute(ship.shape.length + other_ship.shape.length):
            if operating_mode is OperatingMode.MONITOR:
                return True
            else: # Constraint mode not implemented for this case as this area should be never reached - basically already a collision
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            VO = construct_velocity_obstacle(ship.states_cr[time_step], other_ship.states_cr[time_step],
                                             ship.shape.length, other_ship.shape.length)

            v_other = velocity_vector(other_ship.states_cr[time_step])
            v_ship = velocity_vector(ship.states_cr[time_step])
            v_AB = relative_velocity_vector(ship.states_cr[time_step],other_ship.states_cr[time_step])
            VO.translate_rotate(v_other,0.0)
            tip_velocity = v_ship + ship.states_cr[time_step].position

            if VO.contains_point(tip_velocity) and np.any(tip_velocity != ship.states_cr[time_step].position) and \
                    np.linalg.norm(v_AB) >= (d_AB / self._traffic_rules_param.get('t_horizon_collision_possible_soon')):
                if operating_mode is OperatingMode.MONITOR:
                    return True
                elif operating_mode is OperatingMode.CONSTRAINT:
                    return [Constraint([ConstraintType.UNSAFE_VELOCITY_SPACE], ConstraintRepresentation.OUTER_BOUNDARY,
                                       ShapeGroup([VO]))]
                else:
                    raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
            else:
                if operating_mode is OperatingMode.MONITOR:
                        return False
                elif operating_mode is OperatingMode.CONSTRAINT:
                    return []  # nothing to do as rule is not applied
                else:
                    raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def collision_inevitable(self, time_step: int, ship: Ship, other_ship: Ship, t_react: float) -> bool:
        """
        Evaluates if a collision is inevitable by checking if a collision is possible and the safe distance is violated
        which means even full braking of both ships won't avoid the collision

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship: other ship
        :param t_react: Reaction time of ego ship [s]
        :return: Boolean indicating satisfaction
        """
        if not self.safe_distance(time_step, ship, other_ship, t_react, OperatingMode.MONITOR) and\
                self.collision_possible(time_step,ship,other_ship):
            return True
        else:
            return False

    def crossing(self, time_step: int, ship: Ship, other_ship: Ship, operating_mode: OperatingMode) -> bool or List[Constraint]:
        """
        Evaluates if the vehicle is in the give-way position of a crossing encounter

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship: other ship
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :return: boolean indicating satisfaction
        """
        # check if predicates for this state are fulfilled
        if self.collision_possible(time_step, ship, other_ship) and \
                PositionPredicateCollection.in_right_sector(time_step,ship_p=ship, ship_k=other_ship,
                                                            head_on_angle=self._traffic_rules_param.get("max_orientation_diff_head_on"),
                                                            overtake_angle=self._traffic_rules_param.get("max_orientation_diff_overtake")) and \
                PositionPredicateCollection.orientation_towards_left(time_step, ship_p=ship, ship_k=other_ship,
                                                                     head_on_angle=self._traffic_rules_param.get("max_orientation_diff_head_on")):
            if operating_mode is OperatingMode.MONITOR:
                return True
            elif operating_mode is OperatingMode.CONSTRAINT:
                 constraints = []
                 constraints.extend([Constraint([ConstraintType.ACCELERATION], ConstraintRepresentation.UPPER, 0)])
                 constraints.extend([Constraint([ConstraintType.UNSAFE_CARTESIAN_AREA], ConstraintRepresentation.OUTER_BOUNDARY,
                                                give_way_constraint(ship.states_cr[time_step], other_ship.states_cr[time_step],
                                                        t_turning= self._traffic_rules_param.get("t_turning"),
                                                        safe_distance=calculate_safe_distance(ship, other_ship, time_step, ship.vehicle_param['t_react']),
                                                        shift_distance=self._traffic_rules_param.get("shift_distance"),length_ego= ship.shape.length,to_right=True))])
                 return constraints # orientation between current and 90 degree to right, no acceleration
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            if operating_mode is OperatingMode.MONITOR:
                return False
            elif operating_mode is OperatingMode.CONSTRAINT:
                return []  # nothing to do as rule is not applied
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def maneuver_crossing(self, time_step: int, ship: Ship, other_ship: Ship, current_crossing: bool = None) -> bool:
        """
        Predicate which is evaluated if the vessel has to make a crossing maneuver

        :param ship: the ship of interest
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if current_crossing is None:
            current_crossing = self.crossing(time_step, ship, other_ship, operating_mode=OperatingMode.MONITOR)

        if current_crossing:
            if time_step - 1 not in ship.time_steps:
                self.time_step_before_crossing = -1
                return False
            if self.time_step_before_crossing == -1:
                self.time_step_before_crossing = time_step - 1
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_crossing = orientation_change
                if self.change_course(self.current_delta_orientation_crossing,self._traffic_rules_param.get("sig_orientation_diff")) and \
                        self.turning_to_starboard(ship.states_cr[self.time_step_before_crossing].orientation, ship.states_cr[time_step].orientation):
                    return True
                else:
                    return False

            else:
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_crossing += orientation_change
                if self.change_course(self.current_delta_orientation_crossing,self._traffic_rules_param.get("sig_orientation_diff")) and \
                        self.turning_to_starboard(ship.states_cr[self.time_step_before_crossing].orientation, ship.states_cr[time_step].orientation):
                    return True
                else:
                    return False
        else:
            self.time_step_before_crossing = -1
            return False

    def keep(self, time_step: int, ship: Ship, other_ship: Ship, operating_mode) -> bool or List[Constraint]:
        """
        Evaluates if the vehicle is in the stand-on position of a crossing encounter

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship: other ships
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :return: boolean indicating satisfaction
        """
        # check if predicates for this state are fulfilled
        rule_checks = []
        if (self.collision_possible(time_step, ship, other_ship) and \
                PositionPredicateCollection.in_left_sector(time_step, ship_p=ship, ship_k=other_ship,
                                   head_on_angle=self._traffic_rules_param.get("max_orientation_diff_head_on"),
                                   overtake_angle=self._traffic_rules_param.get("max_orientation_diff_overtake")) and \
                PositionPredicateCollection.orientation_towards_right(time_step, ship_p=ship, ship_k=other_ship,
                                                                      head_on_angle=self._traffic_rules_param.get("max_orientation_diff_head_on"))) or \
                self.overtake(time_step,ship=other_ship,other_ship=ship,operating_mode= OperatingMode.MONITOR):
            if operating_mode is OperatingMode.MONITOR:
                return True
            elif operating_mode is OperatingMode.CONSTRAINT:
                constraints = []
                constraints.extend([Constraint([ConstraintType.ACCELERATION], ConstraintRepresentation.UPPER, 0),
                                    Constraint([ConstraintType.ACCELERATION], ConstraintRepresentation.LOWER, 0)])
                constraints.extend(
                    [Constraint([ConstraintType.UNSAFE_CARTESIAN_AREA], ConstraintRepresentation.OUTER_BOUNDARY,
                                stand_on_constraint(ship.states_cr[time_step], ship.shape.length, self._traffic_rules_param.get("shift_distance"),
                                                    self._traffic_rules_param.get("t_plan_horizon")))])
                return constraints
                        # keep same speed (a=0) and do not change the orientation - minimal deviation allowed (5 degree)
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            if operating_mode is OperatingMode.MONITOR:
                return False
            elif operating_mode is OperatingMode.CONSTRAINT:
                return []  # nothing to do as rule is not applied
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def no_turning(self, time_step: int, ship: Ship, other_ship: Ship, current_keep: bool = None) -> bool:
        """
        Predicate which is evaluated if the vessel has to keep its course

        :param ship: the ship of interest
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if current_keep is None:
            current_keep = self.keep(time_step, ship, other_ship, operating_mode=OperatingMode.MONITOR)

        if current_keep:
            if time_step - 1 not in ship.time_steps:
                self.time_step_before_keep = -1
                return True
            if self.time_step_before_keep == -1:
                self.time_step_before_keep = time_step - 1
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_keep = orientation_change
                if not self.change_course(self.current_delta_orientation_keep,self._traffic_rules_param.get("max_orientation_diff_no_change")):
                    return True
                else:
                    return False

            else:
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_keep += orientation_change
                if not self.change_course(self.current_delta_orientation_keep,self._traffic_rules_param.get("max_orientation_diff_no_change")):
                    return True
                else:
                    return False
        else:
            self.time_step_before_keep = -1
            return True

    def overtake(self, time_step: int, ship: Ship, other_ship: Ship, operating_mode: OperatingMode) -> bool or List[Constraint]:
        """
        Evaluates if the vehicle has to overtake another in order to avoid collisions

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship: other ship
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :return: boolean indicating satisfaction
        """

        if self.collision_possible(time_step, ship, other_ship) and \
                PositionPredicateCollection.in_behind_sector(time_step, ship_p=ship, ship_k=other_ship,
                                                             overtake_angle= self._traffic_rules_param.get("max_orientation_diff_overtake")) and \
                VelocityPredicateCollection.drives_faster(time_step, ship, other_ship) and \
                not PositionPredicateCollection.orientation_delta(time_step, ship, other_ship,self._traffic_rules_param.get("max_orientation_diff_overtake")):

            if operating_mode is OperatingMode.MONITOR:
                return True
            elif operating_mode is OperatingMode.CONSTRAINT:
                overtaken_ship = other_ship
                if overtaken_ship.states_cr[time_step].orientation < ship.states_cr[time_step].orientation:
                    # other ship is more oriented to the right, overtake on left side and do not decelerate
                    # theta_min = vector_angle(overtaken_ship.states_cr[time_step].position - ship.states_cr[time_step].position)
                    constraints = []
                    constraints.extend([Constraint([ConstraintType.ACCELERATION], ConstraintRepresentation.LOWER, 0)])
                    constraints.extend([Constraint([ConstraintType.UNSAFE_CARTESIAN_AREA], ConstraintRepresentation.OUTER_BOUNDARY,
                                                   give_way_constraint(ship.states_cr[time_step], other_ship.states_cr[time_step],
                                                           t_turning=self._traffic_rules_param.get("t_turning"),
                                                           safe_distance=calculate_safe_distance(ship, other_ship,time_step, ship.vehicle_param['t_react']),
                                                           shift_distance=self._traffic_rules_param.get("shift_distance"),length_ego= ship.shape.length, to_right=False))])
                    return constraints
                else:
                    # other ship is more oriented to the left, overtake on right side and do not decelerate
                    #theta_max = vector_angle(overtaken_ship.states_cr[time_step].position - ship.states_cr[time_step].position)
                    constraints = []
                    constraints.extend([Constraint([ConstraintType.ACCELERATION], ConstraintRepresentation.LOWER, 0)])
                    constraints.extend([Constraint([ConstraintType.UNSAFE_CARTESIAN_AREA], ConstraintRepresentation.OUTER_BOUNDARY,
                                                   give_way_constraint(ship.states_cr[time_step], other_ship.states_cr[time_step],
                                                           t_turning=self._traffic_rules_param.get("t_turning"),
                                                           safe_distance=calculate_safe_distance(ship, other_ship,time_step,ship.vehicle_param['t_react']),
                                                           shift_distance=self._traffic_rules_param.get("shift_distance"),
                                                           length_ego= ship.shape.length, to_right=True))])
                    return constraints
            # keep same speed (a=0) and do not change the orientation - minimal deviation allowed (5 degree)
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            if operating_mode is OperatingMode.MONITOR:
                return False
            elif operating_mode is OperatingMode.CONSTRAINT:
                return []  # nothing to do as rule is not applied
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def maneuver_overtake(self, time_step: int, ship: Ship, other_ship: Ship, current_overtake: bool = None) -> bool:
        """
        Predicate which is evaluated if the vessel has to make a overtake maneuver

        :param ship: the ship of interest
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if current_overtake is None:
            current_overtake = self.overtake(time_step, ship, other_ship, operating_mode=OperatingMode.MONITOR)

        if current_overtake:
            if time_step - 1 not in ship.time_steps:
                self.time_step_before_overtake = -1
                return False
            if self.time_step_before_overtake == -1:
                self.time_step_before_overtake = time_step - 1
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_overtake = orientation_change
                if self.change_course(self.current_delta_orientation_overtake,self._traffic_rules_param.get("sig_orientation_diff")):
                    return True
                else:
                    return False

            else:
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_overtake += orientation_change
                if self.change_course(self.current_delta_orientation_overtake,self._traffic_rules_param.get("sig_orientation_diff")):
                    return True
                else:
                    return False
        else:
            self.time_step_before_overtake = -1
            return False

    def head_on(self, time_step: int, ship: Ship, other_ship: Ship, operating_mode: OperatingMode) -> bool or List[Constraint]:
        """
        Evaluates if the vehicle is encountering another ship head-on

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship: other ship
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :return: boolean indicating satisfaction
        """
        if self.collision_possible(time_step, ship, other_ship)and \
                PositionPredicateCollection.in_front_sector(time_step, ship_p=ship, ship_k=other_ship,
                                                            head_on_angle=self._traffic_rules_param.get("max_orientation_diff_head_on")) and \
                not PositionPredicateCollection.orientation_delta(time_step, ship, other_ship,self._traffic_rules_param.get("max_orientation_diff_head_on"), offset=np.pi):

            if operating_mode is OperatingMode.MONITOR:
                return True
            elif operating_mode is OperatingMode.CONSTRAINT:
                constraints = []
                constraints.extend([Constraint([ConstraintType.UNSAFE_CARTESIAN_AREA], ConstraintRepresentation.OUTER_BOUNDARY,
                                               give_way_constraint(ship.states_cr[time_step], other_ship.states_cr[time_step],
                                                       t_turning=self._traffic_rules_param.get("t_turning"),
                                                       safe_distance=calculate_safe_distance(ship, other_ship,time_step, ship.vehicle_param['t_react']),
                                                       shift_distance=self._traffic_rules_param.get("shift_distance"),
                                                       length_ego= ship.shape.length, to_right=True))])
                return constraints
                        # orientation between current and 90 degree to right
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')
        else:
            if operating_mode is OperatingMode.MONITOR:
                return False
            elif operating_mode is OperatingMode.CONSTRAINT:
                return []  # nothing to do as rule is not applied
            else:
                raise ValueError('<Monitor> ERROR: Operating mode chosen not implemented.')

    def maneuver_head_on(self, time_step: int, ship: Ship, other_ship: Ship, current_head_on: bool = None) -> bool:
        """
        Predicate which is evaluated if the vessel has to make a head_on maneuver

        :param ship: the ship of interest
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if current_head_on is None:
            current_head_on = self.head_on(time_step, ship, other_ship, operating_mode=OperatingMode.MONITOR)

        if current_head_on:
            if time_step - 1 not in ship.time_steps:
                self.time_step_before_head_on = -1
                return False
            if self.time_step_before_head_on == -1:
                self.time_step_before_head_on = time_step - 1
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_head_on = orientation_change
                if self.change_course(self.current_delta_orientation_head_on,self._traffic_rules_param.get("sig_orientation_diff")) and \
                        self.turning_to_starboard(ship.states_cr[self.time_step_before_head_on].orientation, ship.states_cr[time_step].orientation):
                    return True
                else:
                    return False

            else:
                orientation_change = signed_modulo((ship.states_cr[time_step - 1].orientation - ship.states_cr[time_step].orientation), 2*np.pi)
                self.current_delta_orientation_head_on += orientation_change
                if self.change_course(self.current_delta_orientation_head_on,self._traffic_rules_param.get("sig_orientation_diff")) and \
                        self.turning_to_starboard(ship.states_cr[self.time_step_before_head_on].orientation, ship.states_cr[time_step].orientation):
                    return True
                else:
                    return False
        else:
            self.time_step_before_head_on = -1
            return False

    def emergency_plan(self, time_step: int, ship: Ship, other_ship: Ship, t_react: float) -> bool:
        """
        Evaluates if the ship has to execute an emergency plan because some other ship cannot perform a collision
        avoidance maneuver when the ego ship stays on course

        :param time_step: time step to evaluate
        :param ship: ship of interest
        :param other_ship:
        :param t_react: Reaction time of ego ship [s]
        :return: boolean indicating satisfaction
        """
        # check if predicates for this state are fulfilled
        rule_checks = []
        if self.collision_inevitable(time_step, ship, other_ship, t_react):
            rule_checks.append(1)
        else:
            rule_checks.append(0)
        if sum(rule_checks) == 0:
            return False
        else:
            return True

    @staticmethod
    def turning_to_starboard(start_angle: float, stop_angle: float) -> float:
        """
        evaluates if the change of heading is in the direction of starboard
        :param start_angle: start angle of maneuver
        :param stop_angle: final angle of maneuver
        """
        start_angle = wrap_angle(start_angle)
        stop_angle = wrap_angle(stop_angle)
        if (start_angle > stop_angle and -np.pi < stop_angle - start_angle < 0) or \
                (start_angle < stop_angle and 2*np.pi > stop_angle - start_angle > np.pi):
            return True
        else:
            return False

    @staticmethod
    def change_course(course_change: float, threshold: float):
        """
        evaluates
        :param course_change: cumulative course change since start of maneuver [rad]
        :param threshold: threshold necessary to identify a change in the course [degree]
        """
        if np.absolute(course_change) >= np.deg2rad(threshold):
            return True
        else:
            return False

    def evaluate_predicates(self, ego_ship: Ship, other_ships: List[Ship],  time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_ship: ego vehicle object containing trajectory and other relevant information
        :param other_ships: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :param operating_mode: operating mode which should be used for evaluation (monitor, constraint, or robustness)
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"safe_distance__x_ego_x_o": {},
                           "collision_possible__x_ego_x_o": {},
                           "collision_possible_soon__x_ego_x_o": {},
                           "collision_inevitable__x_ego_x_o": {},
                           "keep__x_ego_x_o": {},
                           "no_turning__x_ego_x_o":{},
                           "crossing__x_ego_x_o": {},
                           "maneuver_crossing__x_ego_x_o": {},
                           "overtake__x_ego_x_o": {},
                           "maneuver_overtake__x_ego_x_o": {},
                           "head_on__x_ego_x_o": {},
                           "maneuver_head_on__x_ego_x_o": {},
                           "emergency_plan__x_ego_x_o": {}
                           }

        for other_ship in other_ships:
            predicate_trace["safe_distance__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["collision_possible__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["collision_possible_soon__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["collision_inevitable__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["keep__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["no_turning__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["crossing__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["maneuver_crossing__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["overtake__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["maneuver_overtake__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["head_on__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["maneuver_head_on__x_ego_x_o"][other_ship.id] = {}
            predicate_trace["emergency_plan__x_ego_x_o"][other_ship.id] = {}
            self.time_step_before_keep = -1
            self.current_delta_orientation_keep = 0
            self.time_step_before_crossing = -1
            self.current_delta_orientation_crossing = 0
            self.time_step_before_overtake = -1
            self.current_delta_orientation_overtake = 0
            self.time_step_before_head_on = -1
            self.current_delta_orientation_head_on = 0

            for time_step in ego_ship.time_steps:
                if time_step not in other_ship.time_steps:
                    continue
                if "keep__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["keep__x_ego_x_o"][other_ship.id][time_step] = \
                        self.keep(time_step, ego_ship, other_ship, operating_mode)
                if "no_turning__x_ego_x_o" in self._necessary_predicates:
                    try:
                        current_keep = predicate_trace["keep__x_ego_x_o"][other_ship.id][time_step]
                        predicate_trace["no_turning__x_ego_x_o"][other_ship.id][time_step] = self.no_turning(time_step,
                                                                                                             ego_ship,
                                                                                                             other_ship, current_keep=current_keep)
                    except KeyError:
                        predicate_trace["no_turning__x_ego_x_o"][other_ship.id][time_step] = self.no_turning(time_step, ego_ship, other_ship)
                if "crossing__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["crossing__x_ego_x_o"][other_ship.id][time_step] = \
                        self.crossing(time_step, ego_ship, other_ship, operating_mode)
                if "maneuver_crossing__x_ego_x_o" in self._necessary_predicates:
                    try:
                        current_crossing = predicate_trace["crossing__x_ego_x_o"][other_ship.id][time_step]
                        predicate_trace["maneuver_crossing__x_ego_x_o"][other_ship.id][time_step] = self.maneuver_crossing(time_step,
                                                                                                             ego_ship,
                                                                                                             other_ship, current_crossing=current_crossing)
                    except KeyError:
                        predicate_trace["maneuver_crossing__x_ego_x_o"][other_ship.id][time_step] = self.maneuver_crossing(time_step, ego_ship, other_ship)
                if "overtake__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["overtake__x_ego_x_o"][other_ship.id][time_step] = \
                        self.overtake(time_step, ego_ship, other_ship, operating_mode)
                if "maneuver_overtake__x_ego_x_o" in self._necessary_predicates:
                    try:
                        current_overtake = predicate_trace["overtake__x_ego_x_o"][other_ship.id][time_step]
                        predicate_trace["maneuver_overtake__x_ego_x_o"][other_ship.id][time_step] = self.maneuver_overtake(time_step,
                                                                                                             ego_ship,
                                                                                                             other_ship, current_overtake=current_overtake)
                    except KeyError:
                        predicate_trace["maneuver_overtake__x_ego_x_o"][other_ship.id][time_step] = self.maneuver_overtake(time_step, ego_ship, other_ship)
                if "head_on__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["head_on__x_ego_x_o"][other_ship.id][time_step] = \
                        self.head_on(time_step, ego_ship, other_ship, operating_mode)
                if "maneuver_head_on__x_ego_x_o" in self._necessary_predicates:
                    try:
                        current_head_on = predicate_trace["head_on__x_ego_x_o"][other_ship.id][time_step]
                        predicate_trace["maneuver_head_on__x_ego_x_o"][other_ship.id][time_step] = self.maneuver_head_on(time_step,
                                                                                                             ego_ship,
                                                                                                             other_ship, current_head_on=current_head_on)
                    except KeyError:
                        predicate_trace["maneuver_head_on__x_ego_x_o"][other_ship.id][time_step] = self.maneuver_head_on(time_step, ego_ship, other_ship)
                if "emergency_plan__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["emergency_plan__x_ego_x_o"][other_ship.id][time_step] = \
                        self.emergency_plan(time_step, ego_ship, other_ship, ego_ship.vehicle_param['t_react'])
                if "safe_distance__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["safe_distance__x_ego_x_o"][other_ship.id][time_step] = \
                        self.safe_distance(time_step, ego_ship, other_ship, ego_ship.vehicle_param['t_react'],
                                           operating_mode)
                if "collision_possible__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["collision_possible__x_ego_x_o"][other_ship.id][time_step] = \
                        self.collision_possible(time_step, ego_ship, other_ship)
                if "collision_possible_soon__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["collision_possible_soon__x_ego_x_o"][other_ship.id][time_step] = \
                        self.collision_possible_soon(time_step, ego_ship, other_ship, operating_mode)
                if "collision_inevitable__x_ego_x_o" in self._necessary_predicates:
                    predicate_trace["collision_inevitable__x_ego_x_o"][other_ship.id][time_step] = \
                        self.collision_inevitable(time_step, ego_ship, other_ship,  ego_ship.vehicle_param['t_react'])

        return predicate_trace

    def evaluate_constraints_at_time_step(self, ego_ship: Ship, other_ships: List[Ship], time_step: int) -> \
            Dict[str, Dict[int, bool]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_step: time step for which the predicates should be evaluated
        :returns dictionary with traces of bool values for each predicate
        """
        constraint_trace = {"collision_possible_soon__x_ego_x_o": {},
                            "keep__x_ego_x_o": {},
                            "crossing__x_ego_x_o": {},
                            "overtake__x_ego_x_o": {},
                            "head_on__x_ego_x_o": {},
                            "all_constraints_at_time_step": []}

        for other_ship in other_ships:
            constraint_trace["collision_possible_soon__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["keep__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["crossing__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["overtake__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["head_on__x_ego_x_o"][other_ship.id] = {}


            if time_step not in other_ship.time_steps:
                continue
            # constraint_trace[time_step] = copy.deepcopy(self._constraint_dict)
            if "collision_possible_soon__x_ego_x_o" in self._necessary_predicates:
                constraint_trace["collision_possible_soon__x_ego_x_o"][other_ship.id] = self.collision_possible_soon(time_step, ego_ship, other_ship, OperatingMode.CONSTRAINT)
                constraint_trace["all_constraints_at_time_step"].extend(
                    constraint_trace["collision_possible_soon__x_ego_x_o"][other_ship.id])
            if "keep__x_ego_x_o" in self._necessary_predicates:
                constraint_trace["keep__x_ego_x_o"][other_ship.id] = self.keep(time_step, ego_ship,
                                                                                          other_ship,
                                                                                          OperatingMode.CONSTRAINT)
                constraint_trace["all_constraints_at_time_step"].extend(
                    constraint_trace["keep__x_ego_x_o"][other_ship.id])
            if "crossing__x_ego_x_o" in self._necessary_predicates:
                constraint_trace["crossing__x_ego_x_o"][other_ship.id] = self.crossing(time_step,
                                                                                                  ego_ship,
                                                                                                  other_ship,
                                                                                                  OperatingMode.CONSTRAINT)
                constraint_trace["all_constraints_at_time_step"].extend(
                    constraint_trace["crossing__x_ego_x_o"][other_ship.id])
            if "overtake__x_ego_x_o" in self._necessary_predicates:
                constraint_trace["overtake__x_ego_x_o"][other_ship.id] = self.overtake(time_step,
                                                                                                  ego_ship,
                                                                                                  other_ship,
                                                                                                  OperatingMode.CONSTRAINT)
                constraint_trace["all_constraints_at_time_step"].extend(
                    constraint_trace["overtake__x_ego_x_o"][other_ship.id])
            if "head_on__x_ego_x_o" in self._necessary_predicates:
                constraint_trace["head_on__x_ego_x_o"][other_ship.id] = self.head_on(time_step, ego_ship,
                                                                                                other_ship,
                                                                                                OperatingMode.CONSTRAINT)
                constraint_trace["all_constraints_at_time_step"].extend(
                    constraint_trace["head_on__x_ego_x_o"][other_ship.id])
        # TODO: whats the rule priority - what about conflicting constraints? - evading maneuvers before stand on?
        return constraint_trace

    def evaluate_constraints(self, ego_ship: Ship, other_ships: List[Ship]) -> \
            Dict[int, Dict[str, float]]:
        """
        Extracts constraints for a vehicle

        :param ego_ship: ego vehicle object containing trajectory and other relevant information
        :param other_ships: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with traces of constraints for each predicate
        """
        constraint_trace = {"collision_possible_soon__x_ego_x_o": {},
                           "keep__x_ego_x_o": {},
                           "crossing__x_ego_x_o": {},
                           "overtake__x_ego_x_o": {},
                           "head_on__x_ego_x_o": {},
                            "all_constraints_at_time_step": {}}
        for time_step in ego_ship.time_steps:
            constraint_trace["all_constraints_at_time_step"][time_step] = []

        for other_ship in other_ships:
            constraint_trace["collision_possible_soon__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["keep__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["crossing__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["overtake__x_ego_x_o"][other_ship.id] = {}
            constraint_trace["head_on__x_ego_x_o"][other_ship.id] = {}

            for time_step in ego_ship.time_steps:

                if time_step not in other_ship.time_steps:
                    continue
                #constraint_trace[time_step] = copy.deepcopy(self._constraint_dict)
                if "collision_possible_soon__x_ego_x_o" in self._necessary_predicates:
                    constraint_trace["collision_possible_soon__x_ego_x_o"][other_ship.id][time_step] = self.collision_possible_soon(time_step, ego_ship, other_ship, OperatingMode.CONSTRAINT)
                    constraint_trace["all_constraints_at_time_step"][time_step].extend(constraint_trace["collision_possible_soon__x_ego_x_o"][other_ship.id][time_step])
                if "keep__x_ego_x_o" in self._necessary_predicates:
                    constraint_trace["keep__x_ego_x_o"][other_ship.id][time_step] = self.keep(time_step, ego_ship, other_ship, OperatingMode.CONSTRAINT)
                    constraint_trace["all_constraints_at_time_step"][time_step].extend(
                        constraint_trace["keep__x_ego_x_o"][other_ship.id][time_step])
                if "crossing__x_ego_x_o" in self._necessary_predicates:
                    constraint_trace["crossing__x_ego_x_o"][other_ship.id][time_step] = self.crossing(time_step, ego_ship, other_ship, OperatingMode.CONSTRAINT)
                    constraint_trace["all_constraints_at_time_step"][time_step].extend(
                        constraint_trace["crossing__x_ego_x_o"][other_ship.id][time_step])
                if "overtake__x_ego_x_o" in self._necessary_predicates:
                    constraint_trace["overtake__x_ego_x_o"][other_ship.id][time_step] = self.overtake(time_step, ego_ship, other_ship, OperatingMode.CONSTRAINT)
                    constraint_trace["all_constraints_at_time_step"][time_step].extend(
                        constraint_trace["overtake__x_ego_x_o"][other_ship.id][time_step])
                if "head_on__x_ego_x_o" in self._necessary_predicates:
                    constraint_trace["head_on__x_ego_x_o"][other_ship.id][time_step] = self.head_on(time_step, ego_ship, other_ship, OperatingMode.CONSTRAINT)
                    constraint_trace["all_constraints_at_time_step"][time_step].extend(
                        constraint_trace["head_on__x_ego_x_o"][other_ship.id][time_step])
        # TODO: whats the rule priority - what about conflicting constraints? - evading maneuvers before stand on?
        return constraint_trace

    def evaluate_robustness(self, ego_vehicle: Ship, other_vehicles: List[Ship]) -> \
            Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts robustness values for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with traces of robustness values for each predicate
        """
        pass