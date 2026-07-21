from typing import List, Dict, Tuple, Set
import mtl


class TrafficRuleMonitorForward:
    """
    Represents single formalized traffic rule
    """
    def __init__(self, logic_formula: Tuple[str, str], vehicle_dependency: bool):
        """
        :param logic_formula: temporal logic formula
        :param vehicle_dependency: boolean indicating if rule must be evaluated with respect to several vehicles
        """
        self._name = logic_formula[0]
        self._logic_formula = logic_formula[1]
        self._monitor = mtl.parse(logic_formula[1])
        self._predicates = self._extract_predicates(logic_formula[1])
        self._vehicle_dependency = vehicle_dependency

    @property
    def name(self) -> str:
        return self._name

    @property
    def predicates(self) -> Set[str]:
        return self._predicates

    @property
    def vehicle_dependency(self) -> bool:
        return self._vehicle_dependency

    @staticmethod
    def _extract_predicates(logic_formula: str) -> Set[str]:
        """
        Extracts all predicates from temporal logic formula given as string

        :param logic_formula: temporal logic formula
        :returns list of predicates
        """
        replacements = ['U', 'X', 'G', 'F', '&', '->', '(', ')', '~', '|', '[', ']', '0', '1', '2', '3', '4', '5',
                        '6', '7', '8', '9', ',', '.']
        for el in replacements:
            logic_formula = logic_formula.replace(el, "")
        predicates_tmp = list(logic_formula.split(" "))
        predicates = [x for x in predicates_tmp if x != ""]
        return set(predicates)

    def evaluate_monitor(self, predicates: Dict[str, List[Tuple[float, bool]]], get_edges: bool = False) -> bool:
        """
        Evaluates monitor with provided trace of predicates

        :param predicates: trace for each predicate used in rule
        :returns boolean indicating if rule is fulfilled
        """
        if get_edges:
            rule_t = []
            if self._monitor(predicates, quantitative=False):
                rule_t.append(-1)
                return rule_t
            else:
                result = self._monitor(predicates, time=None, quantitative=False)
                t_temp = True
                if not result[0][1]:
                    rule_t.append(0)
                    t_temp = False
                for t in result:
                    if t[1] != t_temp:
                        rule_t.append(t[0])
                    t_temp = t[1]
                return rule_t
        else:
            return self._monitor(predicates, quantitative=False)
