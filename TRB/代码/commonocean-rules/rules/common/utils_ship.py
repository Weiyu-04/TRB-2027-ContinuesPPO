import numpy as np
from typing import Dict, List, Tuple
from rules.common.ship import Ship, VehicleClassification
from rules.predicates.predicate_collection_ship import Constraint, ConstraintType, ConstraintRepresentation
from commonroad.geometry.shape import Polygon, ShapeGroup
from commonocean.scenario.state import PMState,YPState,TFState
from commonocean.scenario.obstacle import DynamicObstacle


def create_constraint_param(constraint_param: Dict) -> Dict:
    """
    Update the theta parameters with rad instead of factor of pi

    :param constraint_param: Parameter dict from config
    :return: updated dict
    """

    constraint_param["theta_max"] = constraint_param["theta_max"] * np.pi
    constraint_param["theta_min"] = constraint_param["theta_min"] * np.pi
    return constraint_param


def update_constraint_param(constraint_param: Dict, update_dict: List[Constraint]) -> Dict:
    """
    Updates the constraints with values from other dict

    :param constraint_param: current constraint dict
    :param update_dict: dict with some updates for the dict
    :return: updated constraint dict
    """
    for constraint in update_dict:
        if constraint.axis[0] is ConstraintType.ACCELERATION:
            if constraint.constraint_representation is ConstraintRepresentation.UPPER:
                constraint_param.update({'a_max': get_new_max(constraint_param['a_max'], constraint.value)})
            elif constraint.constraint_representation is ConstraintRepresentation.LOWER:
                constraint_param.update({'a_min': get_new_min(constraint_param['a_min'], constraint.value)})
        if constraint.axis[0] is ConstraintType.VELOCITY:
            if constraint.constraint_representation is ConstraintRepresentation.UPPER:
                constraint_param.update({'v_max': get_new_max(constraint_param['v_max'], constraint.value)})
            elif constraint.constraint_representation is ConstraintRepresentation.LOWER:
                constraint_param.update({'v_min': get_new_min(constraint_param['v_min'], constraint.value)})
        if constraint.axis[0] is ConstraintType.ORIENTATION:
            if constraint.constraint_representation is ConstraintRepresentation.UPPER:
                constraint_param.update({'theta_max': get_new_max(constraint_param['theta_max'], constraint.value)})
            elif constraint.constraint_representation is ConstraintRepresentation.LOWER:
                constraint_param.update({'theta_min': get_new_min(constraint_param['theta_min'], constraint.value)})
        if constraint.axis[0] is ConstraintType.ANGULAR_VELOCITY:
            if constraint.constraint_representation is ConstraintRepresentation.UPPER:
                constraint_param.update({'theta_dot_max': get_new_max(constraint_param['theta_dot_max'], constraint.value)})
            elif constraint.constraint_representation is ConstraintRepresentation.LOWER:
                constraint_param.update({'theta_dot_min': get_new_min(constraint_param['theta_dot_min'], constraint.value)})
        if constraint.axis[0] is ConstraintType.SAFE_DISTANCE:
            if constraint.constraint_representation is ConstraintRepresentation.UPPER:
                constraint_param.update({'d_safe': get_new_max(constraint_param['d_safe'], constraint.value)})

    return constraint_param


def get_new_min(x_min: float, x_potential_update: float) -> float:
    """
    outputs the upper bound of two values

    :param x_min: original value
    :param x_potential_update: potential update
    :return: upper bound
    """
    if x_min > x_potential_update:
        return x_min
    else:
        return x_potential_update


def get_new_max(x_max: float, x_potential_update: float) -> float:
    """
    outputs the lower bound of two values

    :param x_max:  original value
    :param x_potential_update: potential update value
    :return: lower bound
    """
    if x_max < x_potential_update:
        return x_max
    else:
        return x_potential_update


def clip_to_range(value, start=0, end=np.pi*2):

    return (value % (end - start)) + start


def give_way_constraint(ego_state: [PMState,YPState,TFState], other_state: [PMState,YPState,TFState], t_turning: float, safe_distance: float, shift_distance: float, length_ego: float, to_right: bool = True) -> ShapeGroup:

    if to_right:
        add_sd_direction = np.array([np.cos(ego_state.orientation - np.pi/2), np.sin(ego_state.orientation - np.pi/2)])
    else:
        add_sd_direction = np.array(
            [np.cos(ego_state.orientation + np.pi / 2), np.sin(ego_state.orientation + np.pi / 2)])
    x_other_sd = other_state.position + (safe_distance * add_sd_direction)
    x_turn = ((ego_state.velocity * t_turning) + length_ego) * np.array([np.cos(ego_state.orientation), np.sin(ego_state.orientation)]) + ego_state.position

    shift_parallel = shift_distance * np.array([np.cos(ego_state.orientation), np.sin(ego_state.orientation)])

    x_close = intersection_between_two_lines(x_turn, ego_state.position, x_turn - x_other_sd, add_sd_direction)

    return ShapeGroup([Polygon(
        np.array([x_turn, x_other_sd, x_other_sd + shift_parallel, x_close + shift_parallel, x_close, x_turn]))])


def stand_on_constraint(ego_state: [PMState,YPState,TFState], length_ego: float, shift_distance: float, t_horizon: float) -> ShapeGroup:

    vector_orientation = np.array([np.cos(ego_state.orientation), np.sin(ego_state.orientation)])
    vector_perpendicular = np.array(
            [np.cos(ego_state.orientation + np.pi / 2), np.sin(ego_state.orientation + np.pi / 2)])
    x_behind = ego_state.position - length_ego * vector_orientation

    shift_parallel = shift_distance * vector_perpendicular

    x_behind_up = (length_ego /2) * vector_perpendicular + x_behind
    x_infront_up = x_behind_up + vector_orientation * ((ego_state.velocity * t_horizon) + 3/2 * length_ego)

    x_behind_low = - (length_ego / 2) * vector_perpendicular + x_behind
    x_infront_low = x_behind_low + vector_orientation * ((ego_state.velocity * t_horizon) + 3/2 * length_ego)

    polygon_1 = Polygon(np.array([x_behind_up, x_infront_up, x_infront_up + shift_parallel, x_behind_up + shift_parallel, x_behind_up]))
    polygon_2 = Polygon(np.array([x_behind_low, x_infront_low, x_infront_low - shift_parallel, x_behind_low - shift_parallel, x_behind_low]))

    return ShapeGroup([polygon_1, polygon_2])

def calculate_safe_distance(ship: Ship, other_ship: Ship, time_step: int, t_react):

    d_safe = 1 / 2 * (ship.states_cr[time_step].velocity ** 2 / np.abs(ship.vehicle_param['a_min'])) + \
             t_react * (ship.states_cr[time_step].velocity + other_ship.states_cr[
        time_step].velocity) + ship.shape.length / 2 + \
             1 / 2 * (other_ship.states_cr[time_step].velocity ** 2 / np.abs(other_ship.vehicle_param['a_min'])) + \
             other_ship.shape.length / 2

    return d_safe

def in_halfspace(point: np.array, reference_point: np.array, angle_line: float, positive: bool):
    """
    Evaluating if the point is in the have space constructed by a reference point and angle

    :param point: point to be check if it is in the halfspace
    :param reference_point: reference point for line separating halfspaces
    :param angle_line: angle of separating line
    :param positive: which side of the half space should be considered
    :return: Boolean which indicates if point lies in in halfspace
    """
    b = np.array([np.cos(angle_line), np.sin(angle_line)])
    if positive:
        a = np.array([np.cos(angle_line + (np.pi / 2)), np.sin(angle_line + (np.pi / 2))])
    else:
        a = np.array([np.cos(angle_line - (np.pi/2)), np.sin(angle_line - (np.pi/2))])

    d = np.linalg.norm(point-reference_point)
    d_vec = point-reference_point
    cos_phi = np.dot(b, d_vec/d)
    if (point[0] - reference_point[0] - (cos_phi * d*b[0])) / a[0] <= 0:
        return True
    else:
        return False


def construct_velocity_obstacle(state_A: [PMState,YPState,TFState], state_B: [PMState,YPState,TFState], length_A: float, length_B: float) -> Polygon:
    """
    Constructs a velocity obstacle for obstacle B with respect to A given the length of these obstacles, the maximum time
    horizon to consider (constant velocity assumed), and the state of the two obstacles

    :param state_A: main or ego vehicle
    :param state_B: obstacle two which a possible collision should be considered
    :param length_A: length of obstacle A
    :param length_B: length of obstacle B
    :return: Polygon which represents the velocity obstacle
    """

    d_AB = np.linalg.norm(state_A.position - state_B.position)
    center_point_AB = (1/2 * (state_A.position - state_B.position)) + state_B.position

    tangent_points = intersection_between_two_circles(state_B.position, center_point_AB, (length_A+length_B), d_AB/2, d_AB/2)
    #tangent_vector_1 = state_A.position - tangent_points[0]
    #tangent_vector_2 = state_A.position - tangent_points[1]

    #vector_AB = (state_B.position - state_A.position)
    #limit_point = vector_AB * ((time_horizon*state_A.velocity)/np.linalg.norm(vector_AB)) + state_A.position
    #limit_vector = np.array([vector_AB[1], -vector_AB[0]])

    #p1 = intersection_between_two_lines(tangent_points[0], limit_point, tangent_vector_1, limit_vector)
    #p2 = intersection_between_two_lines(tangent_points[1], limit_point, tangent_vector_2, limit_vector)

    return Polygon(np.array([state_A.position, tangent_points[0], state_B.position, tangent_points[1], state_A.position]))


def relative_velocity_vector(state_A: [PMState,YPState,TFState], state_B: [PMState,YPState,TFState]) -> np.array:
    """
    calculating the relative velocity vector between state A and B

    :param state_A: state of own ship
    :param state_B: state of other ship
    :return: np.array of the relative velocity
    """

    v_A = state_A.velocity * np.array([np.cos(state_A.orientation), np.sin(state_A.orientation)])
    v_B = state_B.velocity * np.array([np.cos(state_B.orientation), np.sin(state_B.orientation)])

    return v_B - v_A


def velocity_vector(state_A: [PMState,YPState,TFState]) -> np.array:
    """
    calculating the relative velocity vector between state A and B

    :param state_A: state of own ship
    :return: np.array of the velocity vector
    """

    v_A = state_A.velocity * np.array([np.cos(state_A.orientation), np.sin(state_A.orientation)])

    return v_A


def intersection_between_two_circles(p1: np.array, p2: np.array, r1: float, r2: float, d: float) -> List[np.array]:
    """
    Calculating the intersection between two circles
    :param p1: center of circle 1
    :param p2: center of circle 2
    :param r1: radius of cicle 1
    :param r2: radius of circle 2
    :param d: euclidean distance between centers
    :return: List of intersection points and None if circles are not intersecting
    """
    if d < 0 or d < np.absolute(r1-r2) or d > (r1+r2):
        print('Warning: Circles are not intersecting')
        return [None]
    else:
        # calculate distance between first center and intersection with connecting line through intersection points on
        # line connecting the two centers of the circles
        a = (r1**2 - r2**2 + d**2) / (2*d)
        # calculate minimum distance between intersection point and line between two centers
        h = np.sqrt(r1**2 - a**2)

        # calculate intersection points
        x1 = p1[0] + ((a/d) * (p2[0]-p1[0])) - ((h/d) * (p2[1]-p1[1]))
        y1 = p1[1] + ((a/d) * (p2[1]-p1[1])) + ((h/d) * (p2[0]-p1[0]))
        x2 = p1[0] + ((a/d) * (p2[0]-p1[0])) + ((h/d) * (p2[1]-p1[1]))
        y2 = p1[1] + ((a/d) * (p2[1]-p1[1])) - ((h/d) * (p2[0]-p1[0]))

        return [np.array([x1,y1]), np.array([x2,y2])]


def intersection_between_two_lines(p1: np.array, p2: np.array, v1: np.array, v2: np.array) -> np.array:
    """
    Calculates the intersection point between two lines which are each defined by a point and vector through it

    :param p1: first point
    :param p2: second point
    :param v1: first vector
    :param v2: second vector
    :return: intersection point
    """

    # parallel vectors
    parallel = False
    if v2[0] == 0 or v2[1] == 0:
        if v2[0] == 0:
            if v1[0] == 0:
                parallel = True
        elif v2[1] == 0:
            if v1[1] == 0:
                parallel = True
        else:
            if v1[0] == 0 and v1[1] == 0:
                parallel = True

    elif v1[0] / v2[0] == v1[1] / v2[1]:
        parallel = True

    if parallel:
        return np.array([None])
    else:
        alpha = (p1[1] - p2[1] + ((p2[0]-p1[0]) * (v1[1]/v1[0]))) / (v2[1]-(v2[0]*(v1[1]/v1[0])))
        point = p2 + alpha * v2
        return point


def create_scenario_vehicles(ego_obstacle: DynamicObstacle, ego_vehicle_param: Dict,
                             other_vehicles_param: Dict,
                             dynamic_obstacles: List[DynamicObstacle]) -> Tuple[Ship, List[Ship]]:
    """
    Creates vehicles object for all obstacles within a CommonOcean scenario given

    :param ego_obstacle: CommonOcean obstacle of ego vehicle
    :param ego_vehicle_param: :param vehicle_param: dictionary with vehicle ego parameters
    :param other_vehicles_param: :param vehicle_param: dictionary with vehicle parameters of other traffic participants
    :param dynamic_obstacles: list containing dynamic obstacles of CommonOcean scenario

    :return: ego vehicle object and list of vehicle objects containing other traffic participants
    """
    other_vehicles = []
    initial_time_step = ego_obstacle.initial_state.time_step
    final_time_step = 1000000
    for obstacle in dynamic_obstacles:
        if obstacle.prediction is None:
            continue
        else:
            if initial_time_step < obstacle.initial_state.time_step:
                initial_time_step = obstacle.initial_state.time_step
            if final_time_step > obstacle.prediction.trajectory.final_state.time_step:
                final_time_step = obstacle.prediction.trajectory.final_state.time_step
    ego_vehicle = create_vehicle(ego_obstacle, ego_vehicle_param, initial_time_step, final_time_step)
    for obs in dynamic_obstacles:
        if obs.obstacle_id == ego_obstacle.obstacle_id or obs.prediction is None \
                or obs.initial_state.time_step > ego_vehicle.state_list_cr[-1].time_step \
                or ego_obstacle.initial_state.time_step > obs.prediction.trajectory.state_list[-1].time_step:
            continue
        vehicle = create_vehicle(obs, other_vehicles_param, initial_time_step, final_time_step, ego_vehicle)
        other_vehicles.append(vehicle)
    return ego_vehicle, other_vehicles

def create_scenario_vehicles_dynamically(ego_obstacle: DynamicObstacle, ego_vehicle_param: Dict,
                             other_vehicles_param: Dict,
                             dynamic_obstacles: List[DynamicObstacle]) -> Tuple[Ship, List[Ship]]:
    """
    Creates vehicles object for all obstacles within a CommonOcean scenario given

    :param ego_obstacle: CommonOcean obstacle of ego vehicle
    :param ego_vehicle_param: :param vehicle_param: dictionary with vehicle ego parameters
    :param other_vehicles_param: :param vehicle_param: dictionary with vehicle parameters of other traffic participants
    :param dynamic_obstacles: list containing dynamic obstacles of CommonOcean scenario

    :return: ego vehicle object and list of vehicle objects containing other traffic participants
    """
    other_vehicles = []
    initial_time_step = ego_obstacle.initial_state.time_step
    if ego_obstacle.prediction is None:
        final_time_step = initial_time_step
    else:
        final_time_step = ego_obstacle.prediction.trajectory.final_state.time_step

    ego_vehicle = create_vehicle(ego_obstacle, ego_vehicle_param, initial_time_step, final_time_step)
    for obs in dynamic_obstacles:
        if obs.obstacle_id == ego_obstacle.obstacle_id or obs.prediction is None \
                or obs.initial_state.time_step > ego_vehicle.state_list_cr[-1].time_step \
                or ego_obstacle.initial_state.time_step > obs.prediction.trajectory.state_list[-1].time_step:
            continue
        vehicle = create_vehicle(obs, other_vehicles_param, initial_time_step, final_time_step, ego_vehicle)
        other_vehicles.append(vehicle)
    return ego_vehicle, other_vehicles


def create_vehicle(obstacle: DynamicObstacle, vehicle_param: Dict, initial_time_step: int, final_time_step: int,
                   ego_vehicle: Ship = None) -> Ship:
    """
    Transforms a CommonOcean obstacle to a vehicle object

    :param obstacle: CommonOcean obstacle
    :param vehicle_param: dictionary with vehicle parameters
    :param ego_vehicle: ego vehicle object (if it exist already) for reference generation
    :return: vehicle object
    """
    if ego_vehicle is None:
        vehicle_classification = VehicleClassification.EGO_SHIP
    # TODO: integrate here more ship calssifications if needed - eg crossing, overtaking, encounter ships
    else:
        vehicle_classification = VehicleClassification.OTHER_SHIP

    if initial_time_step == obstacle.initial_state.time_step:
        state_list_cr = {obstacle.initial_state.time_step: obstacle.initial_state}
    else:
        if initial_time_step < obstacle.initial_state.time_step:
            state_list_cr = {obstacle.initial_state.time_step: obstacle.initial_state}
        else:
            state_list_cr = {initial_time_step: obstacle.prediction.trajectory.state_at_time_step(initial_time_step)}

    vehicle_classifications = {initial_time_step: vehicle_classification}
    if obstacle.prediction is not None:
        for state in obstacle.prediction.trajectory.state_list:
            if state.time_step <= initial_time_step or state.time_step > final_time_step:
                continue
            else:
                state_list_cr[state.time_step] = state
                vehicle_classifications[state.time_step] = vehicle_classification

    vehicle = Ship(obstacle.obstacle_shape, state_list_cr, obstacle.obstacle_id, obstacle.obstacle_type, vehicle_param,
                   vehicle_classifications)
    return vehicle


def wrap_angle(angle: float) -> float:
    """
    wrap radian angle to specific range

    :param angle: angle of interest [rad]
    :return: wrapped angle
    """
    min = 0.0
    period = 2*np.pi
    while angle < 0.0:
        angle += period
    if angle < min or angle > min + period:
        wrapped_angle = (angle % period) + min
        return wrapped_angle
    else:
        return angle


def vector_angle(vector: np.array) -> float:

    if vector[0] != 0:
        theta = np.arctan(vector[1]/vector[0])
    else:
        theta = np.arctan(0)

    if vector[0] >= 0:
        if vector[1] >= 0:
            return theta
        else:
            return (2*np.pi) + theta
    else:
        return np.pi + theta


def check_rotation_direction(start: float, stop: float) -> float:
    if (start > stop and stop - start < 0) or (start < stop and stop - start > np.pi):
        return -1
    else:
        return 1


def signed_modulo(nominator: float, divisor: float):
    return np.sign(nominator) * (np.abs(nominator) % np.abs(divisor))
