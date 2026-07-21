import cvxpy as cp
import numpy as np
import torch as th
from cvxpylayers.torch import CvxpyLayer
from scipy.spatial import ConvexHull
from torch import Tensor

from serl_sprl.sets import Zonotope

NUM_GENERATORS = 30
CURRENT_STATE = th.tensor([6.8111077e-02, 1.4874400e00, 1.9564632e-02, -1.0456067e-01, -1.2075857e-03, -1.1258519e00])
NUM_NEURONS = 256
PENALTY_FACTOR = 0.4


def plot_arrow(ax, start_point, normal, length=0.3, label=None, color="red", arrow_width=0.02):
    """
    Plot an arrow from a starting point in the direction of the normal vector.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    start_point : np.ndarray
        Starting point of the arrow (shape: (2,)).
    normal : np.ndarray
        Normal direction vector (shape: (2,)).
    length : float, optional
        Length of the arrow. Default is 0.3.
    label : str, optional
        Label for the arrow. Default is None.
    color : str, optional
        Color of the arrow. Default is 'red'.
    arrow_width : float, optional
        Width of the arrow head. Default is 0.02.
    """
    # Calculate end point
    end_point = start_point + length * normal

    # Plot the arrow
    ax.annotate(
        "",
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(arrowstyle="->", color=color, lw=2, mutation_scale=20, shrinkA=0, shrinkB=0),
        zorder=3,
    )

    # Add label if provided (place it at the end of the arrow)
    if label is not None:
        # Offset the label slightly from the arrow tip
        label_offset = 0.5 * normal
        label_pos = end_point + label_offset
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            color=color,
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            zorder=4,
        )


def plot_shape(ax, vertices, label, color):
    hull = ConvexHull(vertices.T)

    for i, simplex in enumerate(hull.simplices):
        ax.plot(
            vertices[0, simplex], vertices[1, simplex], lw=4, label=label if i == 0 else None, color=color, zorder=2
        )


def dynamics_quadrotor(state, actions, env):
    batch_size = actions.shape[0]
    if not state.shape[0] == actions.shape[0]:
        state = state.repeat(batch_size, 1)  # batch size
    w_max = env.envs[0].get_wrapper_attr("w")
    w = np.random.uniform(-w_max, w_max, size=(2,))
    w = th.tensor(w, dtype=th.float32).repeat(batch_size, 1)
    A_d = th.tensor(env.envs[0].get_wrapper_attr("A_d"), dtype=th.float32)
    A_d = A_d.repeat(batch_size, 1, 1)
    B_d = th.tensor(env.envs[0].get_wrapper_attr("B_d"), dtype=th.float32)
    B_d = B_d.repeat(batch_size, 1, 1)
    E_d = th.tensor(env.envs[0].get_wrapper_attr("E_d"), dtype=th.float32)
    E_d = E_d.repeat(batch_size, 1, 1)
    x_eq = th.tensor(env.envs[0].get_wrapper_attr("x_eq"), dtype=th.float32)
    u_eq = th.tensor(env.envs[0].get_wrapper_attr("u_eq"), dtype=th.float32)
    w_eq = th.tensor(env.envs[0].get_wrapper_attr("w_eq"), dtype=th.float32)
    next_states = (
        x_eq
        + (A_d @ (state - x_eq).unsqueeze(-1)).squeeze(-1)
        + (B_d @ (actions - u_eq).unsqueeze(-1)).squeeze(-1)
        + (E_d @ (w - w_eq).unsqueeze(-1)).squeeze(-1)
    )
    return next_states


def reward_quadrotor(state, action):
    goal_state = th.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    step_reward = -1
    reward_action_weight = 0.01
    reward_state_weight = 1.0
    action_low = th.tensor([6.834830643179076, 6.834830643179076])
    action_high = th.tensor([8.596630030978226, 8.596630030978226])
    action_range = th.stack([action_low, action_high])

    reward = step_reward
    distance = th.linalg.vector_norm(state - goal_state, dim=1)
    dist = reward_state_weight * distance
    act_cost = th.mean((action - action_range[0] / action_range[1] - action_range[0]), dim=1)
    dist += reward_action_weight * act_cost
    reward += th.exp(-dist)
    return reward


def reward_quadrotor_cvxpy(state, action):
    """
    CVXPy-compatible version of the quadrotor reward function.
    Uses DCP-compliant formulation by avoiding square root and using sum of squares directly.
    """
    goal_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    reward_action_weight = 0.01
    reward_state_weight = 1.0
    action_low = np.array([6.834830643179076, 6.834830643179076])
    action_high = np.array([8.596630030978226, 8.596630030978226])

    # Use sum of squares directly (DCP compliant) instead of sqrt(sum_squares)
    distance_squared = cp.sum_squares(state - goal_state)
    dist = reward_state_weight * distance_squared  # Remove sqrt to make it DCP

    # Action cost (fixed the normalization)
    action_normalized = (action - action_low) / (action_high - action_low)
    act_cost = cp.sum(action_normalized) / action.size

    dist += reward_action_weight * act_cost

    # Use quadratic form which is DCP compliant for minimization
    reward_approx = -dist

    return reward_approx


def construct_projection_layer(z):
    action = cp.Parameter(2)

    safe_action = cp.Variable(2)

    objective = cp.Minimize(cp.sum_squares(action - safe_action))

    constraints = Zonotope.point_containment_constraints(
        safe_action,
        z.c.reshape(
            -1,
        ),
        z.G,
    )

    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[action], variables=[safe_action])


def create_layer_safe_action_set(c_omega, G_omega):
    u_rl = cp.Parameter(c_omega.shape[0])
    u = cp.Variable(c_omega.shape[0])
    beta = cp.Variable(G_omega.shape[1])
    G_safe = cp.Parameter(G_omega.shape)
    c_safe = cp.Parameter(c_omega.shape[0])

    constraints = [
        u == c_safe + G_safe @ beta,
        cp.norm(beta, "inf") <= 1,
    ]

    objective = cp.Minimize(cp.square(cp.norm(u - u_rl, 2)))

    prob = cp.Problem(objective, constraints)
    return CvxpyLayer(prob, parameters=[u_rl, c_safe, G_safe], variables=[u, beta])


def compute_safe_set(num_generators, proj_config, env, current_state):
    direction = np.random.rand(2, num_generators) * 2 - 1
    direction = direction / np.linalg.norm(direction, axis=0, keepdims=True)
    # we want to finde the zonotope with max. volume that fulfills our constraints
    center = cp.Variable(2)
    length = cp.Variable(num_generators, nonneg=True)
    generator = direction @ cp.diag(length)
    # compute one step reachable state set
    noise_set_zono = env.envs[0].get_wrapper_attr("safeguard")._noise_set
    noise_center = noise_set_zono.c.reshape((-1, 1))
    noise_generator = noise_set_zono.G
    state_mat = env.envs[0].get_wrapper_attr("A_d")
    action_mat = env.envs[0].get_wrapper_attr("B_d")
    lin_state = env.envs[0].get_wrapper_attr("x_eq")
    lin_action = env.envs[0].get_wrapper_attr("u_eq")
    u_range = env.envs[0].get_wrapper_attr("u_range").transpose()
    input_zono = Zonotope.from_interval(u_range)
    next_state_center = (
        lin_state + state_mat @ (current_state.numpy() - lin_state) + action_mat @ (center - lin_action) + noise_center
    )
    next_state_generator = cp.hstack([action_mat @ generator, noise_generator])

    # find safe action set
    objective = cp.Maximize(cp.geo_mean(length))

    safe_region = env.envs[0].get_wrapper_attr("safe_region")
    state_safety = Zonotope.zonotope_containment_constraints(
        next_state_center, next_state_generator, safe_region.c_S, safe_region.G_S
    )

    action_safety = Zonotope.zonotope_containment_constraints(
        center, generator, input_zono.c.reshape((-1,)), input_zono.G
    )
    constraints = state_safety + action_safety

    problem = cp.Problem(objective, constraints)

    problem.solve()

    z = Zonotope(th.from_numpy(generator.value).type(th.float32), th.from_numpy(center.value).type(th.float32))
    return z


def gaussian_ray_integral_torch(
    mu: Tensor, Sigma: Tensor, start_point: Tensor, direction: Tensor, t_max: float = 10.0
) -> Tensor:
    """
    Compute the line integral of a multivariate Gaussian distribution along a ray using Pyth.

    Parameters:
    -----------
    mu : Tensor, shape (..., d)
        Mean vector(s) of the Gaussian distribution
    Sigma : Tensor, shape (..., d, d)
        Covariance matrix/matrices of the Gaussian distribution
    start_point : Tensor, shape (..., d)
        Starting point(s) of the ray
    direction : Tensor, shape (..., d)
        Direction vector(s) of the ray (will be normalized)
    t_max : float
        Maximum parameter value for integration (ray goes from t=0 to t=t_max)

    Returns:
    --------
    integral : Tensor, shape (...)
        The value of the line integral ∫_0^{t_max} f(start_point + t*direction) dt
    """

    # Ensure tensors are on the same device
    device = mu.device
    mu = mu.to(device)
    Sigma = Sigma.to(device)
    start_point = start_point.to(device)
    direction = direction.to(device)

    # Get dimensions
    d = mu.shape[-1]

    # Compute the inverse and determinant of Sigma
    Sigma_inv = th.linalg.inv(Sigma)  # shape (..., d, d)
    det_Sigma = th.linalg.det(Sigma)  # shape (...)

    # Normalization constant for multivariate Gaussian
    norm_const = 1.0 / ((2 * th.pi) ** (d / 2) * th.sqrt(det_Sigma))  # shape (...)

    # The ray is parameterized as: u(t) = start_point + t * direction
    # We need to integrate: ∫ f(u(t)) dt from 0 to t_max
    # where f(u) = norm_const * exp(-0.5 * (u - mu)^T Sigma^{-1} (u - mu))

    # Express the quadratic form in terms of t:
    # u(t) - mu = (start_point - mu) + t * direction
    # Let v = start_point - mu, d = direction
    # Then (u(t) - mu)^T Sigma^{-1} (u(t) - mu) = (v + t*d)^T Sigma^{-1} (v + t*d)
    #                                             = v^T Sigma^{-1} v + 2*t*v^T Sigma^{-1} d + t^2*d^T Sigma^{-1} d

    v = start_point - mu  # shape (..., d)

    # Compute coefficients
    # A = d^T Sigma^{-1} d
    A = th.sum(direction.unsqueeze(-2) @ Sigma_inv @ direction.unsqueeze(-1), dim=(-2, -1))  # shape (...)

    # B = 2 * v^T Sigma^{-1} d
    B = 2 * th.sum(v.unsqueeze(-2) @ Sigma_inv @ direction.unsqueeze(-1), dim=(-2, -1))  # shape (...)

    # C = v^T Sigma^{-1} v
    C = th.sum(v.unsqueeze(-2) @ Sigma_inv @ v.unsqueeze(-1), dim=(-2, -1))  # shape (...)

    # Complete the square: A*t^2 + B*t + C = A*(t - t_star)^2 + C_star
    # Check that A > 0 (should always be true for positive definite Sigma)
    if not th.all(A > 0):
        raise ValueError("Non-positive definite direction term - check covariance matrix")

    t_star = -B / (2 * A)  # shape (...)
    C_star = C - B**2 / (4 * A)  # shape (...)

    # The integral becomes:
    # norm_const * exp(-C_star/2) * ∫_0^{t_max} exp(-A/2 * (t - t_star)^2) dt

    # Substitute w = sqrt(A/2) * (t - t_star), dw = sqrt(A/2) dt
    sqrt_A_over_2 = th.sqrt(A / 2)  # shape (...)
    w0 = sqrt_A_over_2 * (0 - t_star)  # shape (...)
    w_max = sqrt_A_over_2 * (t_max - t_star)  # shape (...)

    # Clamp w values to prevent saturation
    w0_clamped = th.clamp(w0, min=-5.0, max=5.0)  # error function saturates beyond this
    w_max_clamped = th.clamp(w_max, min=-5.0, max=5.0)

    erf_diff = th.erf(w_max_clamped) - th.erf(w0_clamped)

    # When the difference is too small, it means the ray doesn't intersect the Gaussian
    # Set a minimum value to maintain gradient flow
    erf_diff = th.clamp(erf_diff, min=1e-8)

    # Also clamp C_star to prevent exp overflow
    exp_term = th.exp(th.clamp(-C_star / 2, min=-50, max=50))

    integral = norm_const * exp_term * th.sqrt(2 / A) * th.sqrt(th.tensor(th.pi, device=A.device)) / 2 * erf_diff

    # Clamp final result
    integral = th.clamp(integral, min=1e-10, max=1e3)
    print("integral.max():", integral.max())
    print("integral.min():", integral.min())

    return integral


def compute_gaussian_boundary_mass_torch(
    mu: Tensor, Sigma: Tensor, boundary_points: Tensor, normals: Tensor, t_max: float = 10.0
) -> Tensor:
    """
    Compute the probability mass accumulated at boundary points for multiple samples.

    Parameters:
    -----------
    mu : Tensor, shape (d,)
        Mean vector of the Gaussian distribution
    Sigma : Tensor, shape (d, d)
        Covariance matrix of the Gaussian distribution
    boundary_points : Tensor, shape (batch_size, d)
        Boundary points
    normals : Tensor, shape (batch_size, d)
        Outward normal vectors at boundary points
    t_max : float
        Maximum integration parameter

    Returns:
    --------
    masses : Tensor, shape (batch_size,)
        Probability masses at each boundary point
    """

    # Expand Sigma to diagonal matrix
    Sigma = th.diag_embed(Sigma)  # (batch_size, d, d)

    # Compute the integral for each boundary point
    masses = gaussian_ray_integral_torch(mu, Sigma, boundary_points, normals, t_max)

    return masses
