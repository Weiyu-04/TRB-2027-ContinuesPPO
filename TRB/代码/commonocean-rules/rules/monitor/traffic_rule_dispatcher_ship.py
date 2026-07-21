from typing import List, Dict, Set, Tuple
import warnings

from rules.monitor.monitor_interface_mtl_forward import TrafficRuleMonitorForward
from rules.predicates.velocity_predicates_ship import VelocityPredicateCollection
from rules.predicates.position_predicates_ship import PositionPredicateCollection
from rules.predicates.general_predicates_ship import GeneralPredicateCollection
from rules.common.ship import Ship
from rules.common.helper import OperatingMode


class TrafficRuleDispatcher:
    """
    Manages the different monitors for each traffic rule
    """
    def __init__(self, traffic_rules_forward: Dict[str, str],
                 traffic_rule_sets: Dict[str, str],
                 simulation_param: Dict, traffic_rule_param: Dict, activated_traffic_rule_sets: List[str],
                 vehicle_dependent_rules: List[str], operating_mode: OperatingMode, constraint_param: Dict, time_step = None):
        """
        Constructor

        :param traffic_rules_forward: dictionary with MTL formulas of traffic rules for forward MTL framework
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rule_param: dictionary with parameters of traffic rule parameters
        :param activated_traffic_rule_sets: set of rules which are activated
        :param vehicle_dependent_rules: set of rules which must be evaluated with respect to several vehicles
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :param constraint_param: dictionary with parameters for constraining the utilizable state space
        """
        self._dt = simulation_param.get("dt")
        self._simulation_param = simulation_param
        self._monitors_forward = self.create_forward_monitors(traffic_rules_forward, traffic_rule_sets,
                                                              activated_traffic_rule_sets,
                                                              vehicle_dependent_rules)
        necessary_predicates = self.extract_necessary_predicates()

        self._velocity_predicates = VelocityPredicateCollection(simulation_param,
                                                                traffic_rule_param, necessary_predicates)
        self._position_predicates = PositionPredicateCollection(simulation_param,
                                                                traffic_rule_param, necessary_predicates)
        self._general_predicates = GeneralPredicateCollection(simulation_param,
                                                              traffic_rule_param, necessary_predicates, constraint_param)
        self._time_step = time_step

    def extract_necessary_predicates(self) -> Set[str]:
        """
        Extracts necessary predicates from all active rules
        """
        predicates = set()
        for monitor in self._monitors_forward:
            for pred in monitor.predicates:
                predicates.add(pred)
        return predicates

    @staticmethod
    def create_forward_monitors(traffic_rules: Dict[str, str], traffic_rule_sets: Dict[str, str],
                                activated_traffic_rule_sets: List[str], vehicle_dependent_rules: List[str]) \
            -> List[TrafficRuleMonitorForward]:
        """
        Initialization of monitor for each MTL rule

        :param traffic_rules: dictionary with traffic rules in temporal logic
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :param activated_traffic_rule_sets: set of rules which are activated
        :param vehicle_dependent_rules: set of rules which must be evaluated with respect to several vehicles
        :returns list of monitors
        """
        monitors = []
        for traffic_rule_set_id in activated_traffic_rule_sets:
            if traffic_rule_set_id.startswith("B_"):
                continue
            for rule in traffic_rule_sets.get(traffic_rule_set_id):
                vehicle_dependency = rule in vehicle_dependent_rules
                monitors.append(TrafficRuleMonitorForward((rule, traffic_rules.get(rule)), vehicle_dependency))

        return monitors


    def evaluate_predicates(self, ego_vehicle: Ship, other_vehicles: List[Ship], time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Calls different predicate classes for evaluation predicate of all predicates

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary containing predicate evaluation
        """
        velocity_predicates = self._velocity_predicates.evaluate_predicates(ego_vehicle, other_vehicles, time_interval, operating_mode)
        position_predicates = self._position_predicates.evaluate_predicates(ego_vehicle, other_vehicles, time_interval, operating_mode)
        general_predicates = self._general_predicates.evaluate_predicates(ego_vehicle, other_vehicles, time_interval, operating_mode)

        combined_predicates = {**velocity_predicates, **position_predicates, **general_predicates}
        return combined_predicates

    def evaluate_trajectory(self, ego_vehicle: Ship, other_vehicles: List[Ship],  time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> Dict[str, bool]:
        """
        Evaluates trajectory using forward and backward framework

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary containing predicate evaluation
        """
        if operating_mode is OperatingMode.MONITOR:
            results_forward = self.evaluate_trajectory_forward(ego_vehicle, other_vehicles, time_interval, operating_mode)
            result = {}
            result.update(results_forward)
        elif operating_mode is OperatingMode.CONSTRAINT:
            result = self.evaluate_predicate_constraints(ego_vehicle, other_vehicles)
        else:
            print('<ERROR: Evaluation MONITOR> Mode for evaluation not implemented')
            result = None
        return result

    def evaluate_predicate_constraints(self, ego_vehicle: Ship, other_vehicles: List[Ship]) -> Dict[str, bool]:
        """
        Evaluates trajectory for traffic rule compliance with forward MTL framework

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns each rule with boolean indicating satisfaction
        """
        if self._time_step is None:
            velocity_predicates = self._velocity_predicates.evaluate_constraints(ego_vehicle, other_vehicles)
            position_predicates = self._position_predicates.evaluate_constraints(ego_vehicle, other_vehicles)
            general_predicates = self._general_predicates.evaluate_constraints(ego_vehicle, other_vehicles)
        else:
            velocity_predicates = self._velocity_predicates.evaluate_constraints_at_time_step(ego_vehicle, other_vehicles, self._time_step)
            position_predicates = self._position_predicates.evaluate_constraints_at_time_step(ego_vehicle, other_vehicles, self._time_step)
            general_predicates = self._general_predicates.evaluate_constraints_at_time_step(ego_vehicle, other_vehicles, self._time_step)

        combined_predicates = {**velocity_predicates, **position_predicates, **general_predicates}
        return combined_predicates

    def evaluate_trajectory_forward(self, ego_vehicle: Ship, other_vehicles: List[Ship], time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> Dict[str, bool]:
        """
        Evaluates trajectory for traffic rule compliance with forward MTL framework

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns each rule with boolean indicating satisfaction
        """
        evaluated_predicates = self.evaluate_predicates(ego_vehicle, other_vehicles, time_interval, operating_mode)
        rule_evaluation = {}
        for rule in self._monitors_forward:
            # evaluate rules which only depend on the ego vehicle and the environment
            if rule.vehicle_dependency is False:
                rule_predicates = {}
                for pred in rule.predicates:
                    trace = []
                    for idx, value in enumerate(evaluated_predicates[pred][ego_vehicle.id].values()):
                        trace.append((idx * self._dt, value))
                    rule_predicates[pred] = trace
                if self._simulation_param['generate_viz']:
                    rule_evaluation[rule.name] = rule.evaluate_monitor(rule_predicates, get_edges=True)
                else:
                    rule_evaluation[rule.name] = rule.evaluate_monitor(rule_predicates)
            else:  # evaluate rules which depend on other vehicles, e.g., safe distance
                rule_predicates = {}
                rule_evaluated = False
                for vehicle in other_vehicles:
                    for pred in rule.predicates:
                        trace = []
                        if len(evaluated_predicates[pred]) > 0 \
                                and evaluated_predicates[pred].get(vehicle.id) is not None:
                            for idx, value in enumerate(evaluated_predicates[pred][vehicle.id].values()):
                                trace.append((idx * self._dt, value))
                        elif len(evaluated_predicates[pred]) > 0 \
                                and evaluated_predicates[pred].get(ego_vehicle.id) is not None:
                            for idx, value in enumerate(evaluated_predicates[pred][ego_vehicle.id].values()):
                                trace.append((idx * self._dt, value))
                        else:
                            warnings.warn("Predicate cannot be found!")
                            break
                        rule_predicates[pred] = trace
                        rule_evaluated = True
                    if self._simulation_param['generate_viz']:
                        rule_evaluation[rule.name + "_veh_" + str(vehicle.id)] = rule.evaluate_monitor(rule_predicates, get_edges=True)
                    else:
                        rule_evaluation[rule.name + "_veh_" + str(vehicle.id)] = rule.evaluate_monitor(rule_predicates)
                if rule_evaluated is False:
                    rule_evaluation[rule.name] = True
        return rule_evaluation


