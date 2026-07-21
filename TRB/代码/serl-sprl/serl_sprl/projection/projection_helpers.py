import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer


def create_problem(G_u, c_u, c_w, G_w_hat, G_omega, c_omega, A_hat, B_hat, u_eq, x_eq, multi_step: bool = False):
    r"""Function responsible for creating the problem in cvxpy and parametrize the
    variables to reduce the running time of the solving part as a whole
    Args:
        the same arguments as the ones required to solve the problem per say, they are:
        G_u, c_u, c_w, G_w_hat, G_omega, c_omega, A_hat, B_hat, and u_eq, x_eq;
        multi_step: whether to consider a multi-step ahead prediction (default: False)
    """

    # Create variable u and parameter u_rl
    u = cp.Variable(c_u.shape[0])
    u_rl = cp.Parameter(c_u.shape[0])

    # Create variables Theta and theta
    theta = cp.Variable(G_omega.shape[1])
    Theta = cp.Variable((G_omega.shape[1], G_w_hat.shape[1]))
    lamb = cp.Variable(G_u.shape[0])
    sigma_lamb = cp.Variable(G_u.shape[0])
    sigma_Theta = cp.Variable((G_omega.shape[1], 1))

    # Create parameters that may change at each call
    x_k = cp.Parameter(A_hat.shape[0])

    if multi_step:  # multi-step ahead prediction
        constraints = [
            sigma_lamb >= np.zeros(sigma_lamb.shape),
            sigma_Theta >= np.zeros(sigma_Theta.shape),
            u == c_u + G_u @ lamb,
            cp.norm(lamb, "inf") <= 1 + sigma_lamb,
            # this is for 2-step ahead prediction, can be extended to n-step ahead prediction
            (A_hat**2 @ (x_k - x_eq))
            + A_hat @ B_hat @ (u - u_eq)
            + (B_hat @ (u - u_eq))
            + c_w
            + A_hat @ c_w
            + x_eq
            + A_hat @ x_eq
            == c_omega - (G_omega @ theta),
            G_w_hat + A_hat @ G_w_hat == (G_omega @ Theta),
            cp.norm(cp.hstack([Theta, cp.reshape(theta, (G_omega.shape[1], 1))]), "inf")
            <= np.ones((G_omega.shape[1], 1)) + sigma_Theta,
            # cp.abs(cp.reshape(theta, (G_omega.shape[1], 1))) + cp.abs(Theta) @ np.ones((G_w_hat.shape[1], 1))
            # <= np.ones((G_omega.shape[1], 1)) + sigma_Theta,
        ]
    else:
        constraints = [
            sigma_lamb >= np.zeros(sigma_lamb.shape),
            sigma_Theta >= np.zeros(sigma_Theta.shape),
            u == c_u + G_u @ lamb,
            cp.norm(lamb, "inf") <= 1 + sigma_lamb,
            (A_hat @ (x_k - x_eq)) + (B_hat @ (u - u_eq)) + c_w + x_eq == c_omega - (G_omega @ theta),
            G_w_hat == (G_omega @ Theta),
            cp.norm(cp.hstack([Theta, cp.reshape(theta, (G_omega.shape[1], 1))]), "inf")
            <= np.ones((G_omega.shape[1], 1)) + sigma_Theta,
            # cp.abs(cp.reshape(theta, (G_omega.shape[1], 1))) + cp.abs(Theta) @ np.ones((G_w_hat.shape[1], 1))
            # <= np.ones((G_omega.shape[1], 1)) + sigma_Theta,
        ]

    objective = cp.Minimize(
        cp.square(cp.norm(u - u_rl, 2))
        + 1e3 * np.ones((1, G_omega.shape[1])) @ sigma_Theta
        + 1e3 * np.ones((1, G_u.shape[1])) @ sigma_lamb
    )

    prob = cp.Problem(objective, constraints)

    # Return the problem, the variable of interest and the parameters (notice that we're not solving anything in this
    # function, only building the basis of the solving procedure)
    return prob, u_rl, x_k, u


def create_problem_safe_action_set(c_omega, G_omega):
    u_rl = cp.Parameter(c_omega.shape[0])
    u = cp.Variable(c_omega.shape[0])
    beta = cp.Variable(G_omega.shape[1])
    G_safe = cp.Parameter(G_omega.shape)
    c_safe = cp.Parameter(c_omega.shape[0])

    constraints = [u == c_safe + G_safe @ beta, cp.norm(beta, "inf") <= 1]

    objective = cp.Minimize(cp.square(cp.norm(u - u_rl, 2)))

    prob = cp.Problem(objective, constraints)
    return prob, u_rl, c_safe, G_safe, u


def create_proj_layer_scaled(config, multi_step: bool = False):
    r"""Function responsible for creating the problem in cvxpy and parametrize the variables to
    reduce the running time of the solving part as a whole
    Args:
        the same arguments as the ones required to solve the problem per say,
        they are: G_u, c_u, c_w, G_w_hat, G_omega, c_omega, A_hat, B_hat, and;
    """

    if "A_hat" in config:
        # projection ensures that the reachable set is contained in the RCI set (Quad, Pendulum)
        layer = _create_layer_safe_state_set(config, multi_step)
    else:
        # projection ensures that the action is in the safe action set (Seeker)
        layer = _create_layer_safe_action_set(config)

    return layer


def _create_layer_safe_state_set(config, multi_step: bool = False):
    G_u = config["G_u"]
    c_u = config["c_u"]
    c_w = config["c_w"]
    G_w_hat = config["G_w_hat"]
    G_omega = config["G_omega"]
    c_omega = config["c_omega"]
    A_hat = config["A_hat"]
    B_hat = config["B_hat"]
    u_eq = config["u_eq"]
    x_eq = config["x_eq"]
    u_low = config["u_low"]
    u_high = config["u_high"]
    # Create variable u and parameter u_rl
    u = cp.Variable(c_u.shape[0])
    u_scaled = cp.Variable(c_u.shape[0])
    u_rl = cp.Parameter(c_u.shape[0])

    # Create variables Theta and theta
    theta = cp.Variable(G_omega.shape[1])
    Theta = cp.Variable((G_omega.shape[1], G_w_hat.shape[1]))
    lamb = cp.Variable(G_u.shape[0])
    sigma_lamb = cp.Variable(G_u.shape[0])
    sigma_Theta = cp.Variable((G_omega.shape[1], 1))

    # Create parameters that may change at each call
    x_k = cp.Parameter(A_hat.shape[0])

    if multi_step:  # multi-step ahead prediction
        constraints = [
            sigma_lamb >= np.zeros(sigma_lamb.shape),
            sigma_Theta >= np.zeros(sigma_Theta.shape),
            u == c_u + G_u @ lamb,
            cp.norm(lamb, "inf") <= 1 + sigma_lamb,
            (A_hat**2 @ (x_k - x_eq))
            + A_hat @ B_hat @ (u - u_eq)
            + (B_hat @ (u - u_eq))
            + c_w
            + A_hat @ c_w
            + x_eq
            + A_hat @ x_eq
            == c_omega - (G_omega @ theta),
            G_w_hat + A_hat @ G_w_hat == (G_omega @ Theta),
            cp.abs(cp.reshape(theta, (G_omega.shape[1], 1))) + cp.abs(Theta) @ np.ones((G_w_hat.shape[1], 1))
            <= np.ones((G_omega.shape[1], 1)) + sigma_Theta,
            # cp.abs(Theta) @ np.ones((G_w_hat.shape[1], 1)) <= np.ones((G_omega.shape[1], 1)),
            u_scaled == 2.0 * ((u - u_low) / (u_high - u_low)) - 1,
        ]
    else:
        constraints = [
            sigma_lamb >= np.zeros(sigma_lamb.shape),
            sigma_Theta >= np.zeros(sigma_Theta.shape),
            u == c_u + G_u @ lamb,
            cp.norm(lamb, "inf") <= 1 + sigma_lamb,
            (A_hat @ (x_k - x_eq)) + (B_hat @ (u - u_eq)) + c_w + x_eq == c_omega - (G_omega @ theta),
            G_w_hat == (G_omega @ Theta),
            cp.abs(cp.reshape(theta, (G_omega.shape[1], 1))) + cp.abs(Theta) @ np.ones((G_w_hat.shape[1], 1))
            <= np.ones((G_omega.shape[1], 1)) + sigma_Theta,
            # cp.abs(Theta) @ np.ones((G_w_hat.shape[1], 1)) <= np.ones((G_omega.shape[1], 1)),
            u_scaled == 2.0 * ((u - u_low) / (u_high - u_low)) - 1,
        ]

    objective = cp.Minimize(
        cp.square(cp.norm(u_scaled - u_rl, 2))
        + 1e4 * np.ones((1, G_omega.shape[1])) @ sigma_Theta
        + 1e4 * np.ones((1, G_u.shape[1])) @ sigma_lamb
    )

    prob = cp.Problem(objective, constraints)

    return CvxpyLayer(
        prob,
        parameters=[u_rl, x_k],
        variables=[u_scaled, u, theta, Theta, lamb, sigma_lamb, sigma_Theta],
    )


def _create_layer_safe_action_set(config):
    c_omega = config["c_omega"]
    G_omega = config["G_omega"]
    u_low = config["u_low"]
    u_high = config["u_high"]

    u_rl = cp.Parameter(c_omega.shape[0])
    u = cp.Variable(c_omega.shape[0])
    u_scaled = cp.Variable(c_omega.shape[0])
    beta = cp.Variable(G_omega.shape[1])
    G_safe = cp.Parameter(G_omega.shape)
    c_safe = cp.Parameter(c_omega.shape[0])

    constraints = [
        u == c_safe + G_safe @ beta,
        cp.norm(beta, "inf") <= 1,
        u_scaled == 2.0 * ((u - u_low) / (u_high - u_low)) - 1,
    ]

    objective = cp.Minimize(cp.square(cp.norm(u - u_rl, 2)))

    prob = cp.Problem(objective, constraints)
    return CvxpyLayer(prob, parameters=[u_rl, c_safe, G_safe], variables=[u_scaled, u, beta])
