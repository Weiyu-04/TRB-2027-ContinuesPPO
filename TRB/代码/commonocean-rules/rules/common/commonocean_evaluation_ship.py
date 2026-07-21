import traceback
from typing import Union, Set, Dict, List, Tuple

from rules.common.helper import load_yaml, create_simulation_param, create_ego_vehicle_param, OperatingMode
from rules.monitor.traffic_rule_dispatcher_ship import TrafficRuleDispatcher

from rules.common.utils_ship import create_constraint_param, create_scenario_vehicles, create_scenario_vehicles_dynamically

from commonocean.scenario.scenario import Scenario


class CommonOceanObstacleEvaluation:
    """Class for the traffic rule evaluation of CommonOcean scenarios"""
    def __init__(self, config_path: str, activated_traffic_rule_set: str = None):
        config = load_yaml(config_path + "config_ship.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules_ship.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 10.0)
        self._other_vehicles_param = config.get("other_vehicles_param")
        self._traffic_rules_param = traffic_rules.get("traffic_rules_param")
        self._ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self._simulation_param)
        self._traffic_rule_sets = traffic_rules.get("traffic_rule_sets")
        self._traffic_rules_forward = traffic_rules.get("traffic_rules_forward")
        if activated_traffic_rule_set is None:
            self._activated_traffic_rule_sets = traffic_rules.get("activated_traffic_rule_sets")
        else:
            self._activated_traffic_rule_sets = activated_traffic_rule_set
        self._vehicle_dependent_rules = traffic_rules.get("vehicle_dependent_rules")
        self._operating_mode = OperatingMode(self._simulation_param["mode"])
        self._constraint_param = create_constraint_param(config.get("constraint_param"))

        self.num_vehicles = 0
        self.num_scenarios = 0
        self.num_veh_all_correct = 0
        self.vehicles_dict = {}
        self.eval_dict, self.eval_vehicle_dependent_rules = self._init_eval_dict()

    @property
    def simulation_param(self) -> Dict:
        return self._simulation_param

    @property
    def ego_vehicle_param(self) -> Dict:
        return self._ego_vehicle_param

    @property
    def activated_traffic_rule_sets(self) -> List[str]:
        return self._activated_traffic_rule_sets

    @activated_traffic_rule_sets.setter
    def activated_traffic_rule_sets(self, activated_traffic_rule_sets: List[str]):
        self._activated_traffic_rule_sets = activated_traffic_rule_sets

    def _execute_evaluation(self, scenario: Scenario, time_step = None) -> List[Tuple[int, Dict[str, bool]]]:
        """
        Traffic rule evaluation of each vehicle in a CommonOcean scenario

        :param scenario: CommonOcean scenario
        :return: evaluation results for each vehicle
        """
        dispatcher = TrafficRuleDispatcher(self._traffic_rules_forward,
                                           self._traffic_rule_sets,
                                           self._simulation_param, self._traffic_rules_param,
                                           self._activated_traffic_rule_sets, self._vehicle_dependent_rules,
                                           self._operating_mode, self._constraint_param, time_step)
        vehicle_evaluation = []
        time_interval = (0,1)
        if self.simulation_param.get("mode") == "monitor":
            operating_mode = OperatingMode.MONITOR
        elif self.simulation_param.get("mode") == "constraint":
            operating_mode = OperatingMode.CONSTRAINT
        for ego in scenario.dynamic_obstacles:
            if not (self.simulation_param.get("operating_mode") == "test"
                    or self.simulation_param.get("operating_mode") == "evaluation"
                    or self.simulation_param.get("operating_mode") == "single_scenario"
                    or self.simulation_param.get("operating_mode") == "single_scenario_limited"
                    or (self.simulation_param.get("operating_mode") == "single_vehicle"
                        and self.simulation_param.get("ego_vehicle_id") == ego.obstacle_id)):
                continue
            if (self.simulation_param.get("operating_mode") == "evaluation"
                or self.simulation_param.get("operating_mode") == "single_scenario_limited") \
                    and self.simulation_param.get("num_vehicles") <= self.num_vehicles + len(vehicle_evaluation):
                break
            if operating_mode == OperatingMode.CONSTRAINT:
                ego_vehicle, other_vehicles = create_scenario_vehicles_dynamically(scenario.obstacle_by_id(ego.obstacle_id),
                                                                       self._ego_vehicle_param,
                                                                       self._other_vehicles_param,
                                                                       scenario.dynamic_obstacles)
            else:
                ego_vehicle, other_vehicles = create_scenario_vehicles(scenario.obstacle_by_id(ego.obstacle_id),
                                                                   self._ego_vehicle_param, self._other_vehicles_param,
                                                                   scenario.dynamic_obstacles)
            if operating_mode is OperatingMode.MONITOR:
                vehicle_evaluation.append((ego_vehicle.id, dispatcher.evaluate_trajectory(ego_vehicle, other_vehicles, time_interval, operating_mode)))
            elif operating_mode is OperatingMode.CONSTRAINT:
                vehicle_evaluation.append((ego_vehicle.id,
                                           dispatcher.evaluate_trajectory(ego_vehicle, other_vehicles, time_interval,
                                                                          operating_mode)))
            else:
                print('<ERROR: Evaluation MONITOR> Mode for evaluation not implemented')
        return vehicle_evaluation

    def evaluate_scenario(self, scenario: Scenario, time_step = None, flag_print = True) \
            -> Union[List[Tuple[int, Dict[str, bool]]], None]:
        """
        Evaluates CommonOcean scenario

        :param scenario: CommonOcean scenario
        :return: evaluation results
        """
        self._simulation_param["dt"] = scenario.dt
        try:
            result = self._execute_evaluation(scenario, time_step= time_step)
        except RuntimeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Runtime Error")
            traceback.print_exc()
            return
        except AttributeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Attribute Error")
            traceback.print_exc()
            return
        except KeyError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Key Error")
            traceback.print_exc()
            return
        except ValueError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Value Error")
            traceback.print_exc()
            return
        except IndexError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Index Error")
            traceback.print_exc()
            return
        if self.simulation_param.get("mode") == "monitor":
            self.evaluate_result(result, scenario.benchmark_id, flag_print)
        elif self.simulation_param.get("mode") == "constraint":
            self.evaluate_constraint(result)
        else:
            print('<ERROR: Evaluation MONITOR> Mode for evaluation not implemented')
        return result

    def update_eval_dict(self):
        self.eval_dict, self.eval_vehicle_dependent_rules = self._init_eval_dict()

    def _init_eval_dict(self) -> Tuple[Dict[str, int], Dict[str, bool]]:
        """
        Prepares and creates evaluation dictionaries

        :return: evaluation results
        """
        eval_dict = {}
        eval_vehicle_dependent_rules = {}
        for traffic_rule_set_id in self._activated_traffic_rule_sets:
            for rule_name in self._traffic_rule_sets.get(traffic_rule_set_id):
                if "_".join(rule_name.split("_", 2)[:2]) in self._vehicle_dependent_rules \
                        and eval_vehicle_dependent_rules.get(rule_name) is None:
                    eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = True
                    eval_dict["_".join(rule_name.split("_", 2)[:2])] = 0
                elif eval_dict.get(rule_name) is None and "_".join(rule_name.split("_", 2)[:2]):
                    eval_dict[rule_name] = 0

        return eval_dict, eval_vehicle_dependent_rules

    def evaluate_result(self, result, scenario_name, flag_print = True):
        """
        Statistical evaluation of results

        :param result: evaluation results
        :param scenario_name: CommonOcean scenario name
        """
        self.num_vehicles += len(result)
        self.num_scenarios += 1
        num_correct_rules = 0
        for vehicle in result:
            out_string = "* Scenario: " + scenario_name + " - evaluated obs-id: " + str(vehicle[0])
            for rule_name, eval_result in vehicle[1].items():
                if "_".join(rule_name.split("_", 2)[:2]) in self._vehicle_dependent_rules:
                    if eval_result is False:
                        self.eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = False
                elif eval_result is True:
                    self.eval_dict[rule_name] += 1
                    num_correct_rules += 1
                    out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
                elif eval_result is False:
                    out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
            for rule_name, eval_result in self.eval_vehicle_dependent_rules.items():
                if eval_result is True:
                    self.eval_dict["_".join(rule_name.split("_", 2)[:2])] += 1
                    num_correct_rules += 1
                out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
                self.eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = True
            if num_correct_rules == len(self.eval_dict.keys()):
                self.num_veh_all_correct += 1
            num_correct_rules = 0
            if flag_print:
                print(out_string)
            else:
                pass

    def evaluate_constraint(self, result):
        pass