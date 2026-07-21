import numpy as np


def circle_contains_point(circle_pos: np.ndarray, circle_radius: float, p: np.ndarray) -> bool:
    """Check if a point is inside a circle."""
    return np.linalg.norm(circle_pos - p) < circle_radius


def circle_overlap(
    circle_pos1: np.ndarray,
    circle_radius1: float,
    circle_pos2: np.ndarray,
    circle_radius2: float,
    safety_distance: float = 0,
) -> bool:
    """Check if circles overlap"""
    distance = np.linalg.norm(circle_pos1 - circle_pos2)
    return distance < (circle_radius1 + circle_radius2 + safety_distance)


def any_circle_overlap(
    circle_pos: np.ndarray,
    circle_radius: float,
    other_pos: list[np.ndarray],
    other_radius: list[float],
    safety_distance: float = 0,
) -> bool:
    """Check if circle overlaps with any of the list of circles"""
    for i in range(len(other_pos)):
        if circle_overlap(circle_pos, circle_radius, other_pos[i], other_radius[i], safety_distance):
            return True
    return False


def get_boundary_distance_from_line(p: np.ndarray, d: np.ndarray, boundary_size: float) -> float:
    """
    For the line p + d * a, compute a for which the line intersects the
    boundary box of size boundary_size.
    """
    a_0 = np.abs((boundary_size - p[0]) / d[0])
    a_1 = np.abs((boundary_size - p[1]) / d[1])

    return min(a_0, a_1)


def order_vertices_clockwise(vertices: np.ndarray) -> np.ndarray:
    """Order the vertices of a polygon in a clockwise manner."""
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(-angles)]
