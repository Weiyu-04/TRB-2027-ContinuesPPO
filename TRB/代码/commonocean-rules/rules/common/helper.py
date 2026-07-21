from typing import Dict, Union
import ruamel.yaml
import yaml

import enum
from decimal import Decimal



@enum.unique
class OperatingMode(enum.Enum):
    MONITOR = "monitor"
    CONSTRAINT = "constraint"
    ROBUSTNESS = "robustness"


def create_ego_vehicle_param(ego_vehicle_param: Dict, simulation_param: Dict) -> Dict:
    """
    Update ego vehicle parameters

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :returns updated dictionary
    """

    if not -1e-12 <= (Decimal(str(ego_vehicle_param.get("t_react"))) %
                      Decimal(str(simulation_param.get("dt")))) <= 1e-12:
        raise ValueError('Reaction time must be multiple of time step size.')

    return ego_vehicle_param


def create_simulation_param(simulation_param: Dict, dt: float) -> Dict:
    """
    Update simulation parameters

    :param simulation_param: dictionary with parameters of the simulation environment
    :param dt: time step size of CommonOcean scenario
    :returns updated dictionary with parameters of CommonOcean scenario
    """
    simulation_param["dt"] = dt

    return simulation_param


def load_yaml(file_name: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    with open(file_name, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None
