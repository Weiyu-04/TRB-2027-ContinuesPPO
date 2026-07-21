from typing import Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from torch import Tensor
from torch.nn import functional as F


def squashed_gaussian_ray_integral_torch(
    mu: Tensor, Sigma: Tensor, start_point: Tensor, direction: Tensor, t_max: float = 10.0, num_samples: int = 1000
) -> Tensor:
    """
    Compute the line integral of a squashed multivariate Gaussian (tanh transform)
    along a ray using numerical integration.

    Parameters:
    -----------
    mu : Tensor, shape (..., d)
        Mean vector(s) of the *underlying* Gaussian distribution X ~ N(mu, Sigma)
    Sigma : Tensor, shape (..., d, d)
        Covariance matrix/matrices of the underlying Gaussian
    start_point : Tensor, shape (..., d)
        Starting point(s) of the ray (in squashed space, must be in (-1,1)^d)
    direction : Tensor, shape (..., d)
        Direction vector(s) of the ray (will be normalized)
    t_max : float
        Maximum parameter value for integration
    num_samples : int
        Number of samples for numerical integration

    Returns:
    --------
    integral : Tensor, shape (...)
        The value of the line integral ∫_0^{t_max} f_Y(start_point + t*direction) dt
        where Y = tanh(X) and f_Y is the PDF of the squashed Gaussian
    """

    device = mu.device
    mu = mu.to(device)
    Sigma = Sigma.to(device)
    start_point = start_point.to(device)
    direction = direction.to(device)

    # Get dimensions
    d = mu.shape[-1]
    batch_shape = mu.shape[:-1]

    # Compute Gaussian normalization and inverse covariance
    Sigma_inv = th.linalg.inv(Sigma)
    det_Sigma = th.linalg.det(Sigma)
    # Clamp determinant to prevent division by very small numbers
    det_Sigma = th.clamp(det_Sigma, min=1e-10)
    norm_const = 1.0 / ((2 * th.pi) ** (d / 2) * th.sqrt(det_Sigma))

    # Determine valid integration range
    # For each component i: -1 < start_point[i] + t * direction[i] < 1
    # This gives: (-1 - start_point[i]) / direction[i] < t < (1 - start_point[i]) / direction[i]
    # We need to find the intersection of all these constraints with [0, t_max]

    t_min_valid = th.zeros(batch_shape, device=device)
    t_max_valid = th.full(batch_shape, t_max, device=device)

    # For each dimension, compute constraints
    for i in range(d):
        sp_i = start_point[..., i]
        dir_i = direction[..., i]

        # Handle direction[i] > 0
        pos_mask = dir_i > 1e-10
        if pos_mask.any():
            t_upper = (1.0 - sp_i) / dir_i
            t_max_valid = th.where(pos_mask, th.minimum(t_max_valid, t_upper), t_max_valid)

        # Handle direction[i] < 0
        neg_mask = dir_i < -1e-10
        if neg_mask.any():
            t_upper = (-1.0 - sp_i) / dir_i
            t_max_valid = th.where(neg_mask, th.minimum(t_max_valid, t_upper), t_max_valid)

    # Clip to ensure valid range
    t_max_valid = th.maximum(t_min_valid, t_max_valid)

    # Create integration points (trapezoidal rule)
    # Shape: (..., num_samples)
    t_vals = th.linspace(0, 1, num_samples, device=device)

    # Expand for broadcasting: (..., num_samples, 1)
    if len(batch_shape) > 0:
        t_vals = t_vals.view(*([1] * len(batch_shape)), num_samples)
    else:
        t_vals = t_vals.view(num_samples)

    # Scale t by the valid range for each batch element
    # t_vals_scaled shape: (..., num_samples)
    t_vals_scaled = t_min_valid.unsqueeze(-1) + t_vals * (t_max_valid - t_min_valid).unsqueeze(-1)

    # Compute points along the ray: y(t) = start_point + t * direction
    # Shape: (..., num_samples, d)
    y_vals = start_point.unsqueeze(-2) + t_vals_scaled.unsqueeze(-1) * direction.unsqueeze(-2)

    # Check bounds and clip (for numerical stability)
    # Use more conservative bounds to prevent Jacobian explosion
    y_vals = th.clamp(y_vals, -0.99, 0.99)

    # Transform to original space: x = arctanh(y)
    # arctanh is numerically unstable near ±1, so we clamp
    x_vals = th.atanh(y_vals)  # Shape: (..., num_samples, d)

    # Compute Gaussian PDF at x_vals
    # (x - mu)^T Sigma^{-1} (x - mu)
    diff = x_vals - mu.unsqueeze(-2)  # Shape: (..., num_samples, d)

    # Quadratic form: (..., num_samples)
    quad_form = th.sum(diff.unsqueeze(-2) @ Sigma_inv.unsqueeze(-3) @ diff.unsqueeze(-1), dim=(-2, -1))
    
    # Clamp quadratic form to prevent exp underflow/overflow
    quad_form = th.clamp(quad_form, max=50.0)

    gaussian_pdf = norm_const.unsqueeze(-1) * th.exp(-0.5 * quad_form)

    # Compute Jacobian: ∏ᵢ (1 / (1 - yᵢ²))
    # Use log-sum-exp trick for numerical stability
    log_jacobian = -th.sum(th.log(1.0 - y_vals**2 + 1e-8), dim=-1)  # Shape: (..., num_samples)
    # Clamp log_jacobian to prevent extreme values
    log_jacobian = th.clamp(log_jacobian, max=10.0)  # exp(10) ≈ 22026, reasonable upper bound
    jacobian = th.exp(log_jacobian)

    # Squashed Gaussian PDF
    squashed_pdf = gaussian_pdf * jacobian

    # Numerical integration using trapezoidal rule
    dt = (t_max_valid - t_min_valid) / (num_samples - 1)

    # Trapezoidal weights: [0.5, 1, 1, ..., 1, 0.5]
    weights = th.ones_like(t_vals_scaled)
    weights[..., 0] = 0.5
    weights[..., -1] = 0.5

    integral = th.sum(squashed_pdf * weights, dim=-1) * dt

    # Clamp final result to prevent unrealistic values
    integral = th.clamp(integral, min=1e-10, max=1e3)

    return integral


def gaussian_ray_integral_torch(
    mu: Tensor, Sigma: Tensor, start_point: Tensor, direction: Tensor, t_max: float = 10.0
) -> Tensor:
    """
    Compute the line integral of a multivariate Gaussian distribution along a ray using PyTorch.

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

    return integral


def compute_gaussian_boundary_mass_torch(
    mu: Tensor,
    Sigma: Tensor,
    boundary_points: Tensor,
    normals: Tensor,
    t_max: float = 10.0,
    squashed: bool = False,
    num_samples: int = 100,
) -> Tensor:
    """
    Compute the probability mass accumulated at boundary points for multiple samples.

    Parameters:
    -----------
    mu : Tensor, shape (d,)
        Mean vector of the Gaussian distribution (in Gaussian space if squashed=True)
    Sigma : Tensor, shape (d, d)
        Covariance matrix of the Gaussian distribution (in Gaussian space if squashed=True)
    boundary_points : Tensor, shape (batch_size, d)
        Boundary points (in squashed space if squashed=True)
    normals : Tensor, shape (batch_size, d)
        Outward normal vectors at boundary points (in squashed space if squashed=True)
    t_max : float
        Maximum integration parameter
    squashed : bool
        Whether the distribution is a squashed Gaussian (tanh-transformed)

    Returns:
    --------
    masses : Tensor, shape (batch_size,)
        Probability masses at each boundary point
    """

    if squashed:
        Sigma = th.diag_embed(Sigma.squeeze(0).expand(boundary_points.shape[0], -1))
        # Expand mu for batch
        mu = mu.squeeze(0).expand(boundary_points.shape[0], -1)

        # Compute the integral for each boundary point
        masses = squashed_gaussian_ray_integral_torch(mu, Sigma, boundary_points, normals, t_max, num_samples)

        return masses
    else:
        # Original computation for non-squashed Gaussian
        # Expand Sigma to diagonal matrix
        Sigma = th.diag_embed(Sigma)  # (batch_size, d, d)

        # Compute the integral for each boundary point
        masses = gaussian_ray_integral_torch(mu, Sigma, boundary_points, normals, t_max)

        return masses


def patch_after_imports():
    """
    Patch preprocess_obs after all modules have been imported.
    Call this right before creating your model.
    """
    import sys

    # Get all loaded modules
    for module_name, module in sys.modules.items():
        if module_name.startswith("stable_baselines3") and hasattr(module, "preprocess_obs"):
            setattr(module, "preprocess_obs", preprocess_obs_float64)
            print(f"Patched preprocess_obs in {module_name}")


def preprocess_obs_float64(
    obs: Union[th.Tensor, dict[str, th.Tensor]],
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, dict[str, th.Tensor]]:
    """
    Custom preprocess_obs that respects the current default dtype instead of forcing float32.
    """
    if isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs_float64(
                _obs, observation_space[key], normalize_images=normalize_images
            )
        return preprocessed_obs

    assert isinstance(obs, th.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.to(dtype=th.get_default_dtype()) / 255.0
        return obs.to(dtype=th.get_default_dtype())  # Use current default dtype instead of .float()

    elif isinstance(observation_space, spaces.Discrete):
        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).to(dtype=th.get_default_dtype())

    elif isinstance(observation_space, spaces.MultiDiscrete):
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).to(dtype=th.get_default_dtype())
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.to(dtype=th.get_default_dtype())
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")


# ==============================================================================
# DTYPE MANAGEMENT FUNCTIONS
# ==============================================================================


def get_numpy_dtype_from_torch(torch_dtype: th.dtype) -> np.dtype:
    """Convert PyTorch dtype to corresponding NumPy dtype."""
    dtype_mapping = {
        th.float32: np.float32,
        th.float64: np.float64,
        th.float16: np.float16,
        th.int32: np.int32,
        th.int64: np.int64,
        th.int16: np.int16,
        th.int8: np.int8,
        th.uint8: np.uint8,
        th.bool: np.bool_,
    }

    return dtype_mapping.get(torch_dtype, np.float32)


def get_torch_dtype_from_numpy(numpy_dtype: np.dtype) -> th.dtype:
    """Convert NumPy dtype to corresponding PyTorch dtype."""
    dtype_mapping = {
        np.float32: th.float32,
        np.float64: th.float64,
        np.float16: th.float16,
        np.int32: th.int32,
        np.int64: th.int64,
        np.int16: th.int16,
        np.int8: th.int8,
        np.uint8: th.uint8,
        np.bool_: th.bool,
    }

    return dtype_mapping.get(numpy_dtype, th.float32)


def get_buffer_dtype() -> np.dtype:
    """Get the appropriate NumPy dtype based on current PyTorch default dtype."""
    return get_numpy_dtype_from_torch(th.get_default_dtype())


def setup_dtype(dtype: Union[str, th.dtype, np.dtype, None] = None) -> tuple[th.dtype, np.dtype]:
    """
    Setup and synchronize dtypes between PyTorch and NumPy.

    Parameters
    ----------
    dtype : Union[str, th.dtype, np.dtype, None]
        Desired dtype. Can be:
        - String: "float32", "float64", etc.
        - PyTorch dtype: th.float32, th.float64, etc.
        - NumPy dtype: np.float32, np.float64, etc.
        - None: Use current PyTorch default

    Returns
    -------
    tuple[th.dtype, np.dtype]
        Tuple of (torch_dtype, numpy_dtype)
    """
    if dtype is None:
        torch_dtype = th.get_default_dtype()
        numpy_dtype = get_numpy_dtype_from_torch(torch_dtype)
    elif isinstance(dtype, str):
        # Handle string dtype specifications
        if dtype == "float32":
            torch_dtype = th.float32
            numpy_dtype = np.float32
        elif dtype == "float64":
            torch_dtype = th.float64
            numpy_dtype = np.float64
        elif dtype == "float16":
            torch_dtype = th.float16
            numpy_dtype = np.float16
        else:
            raise ValueError(f"Unsupported dtype string: {dtype}")

        # Set PyTorch default
        th.set_default_dtype(torch_dtype)

    elif isinstance(dtype, th.dtype):
        torch_dtype = dtype
        numpy_dtype = get_numpy_dtype_from_torch(dtype)
        th.set_default_dtype(torch_dtype)

    elif isinstance(dtype, np.dtype) or isinstance(dtype, type):
        numpy_dtype = np.dtype(dtype)
        torch_dtype = get_torch_dtype_from_numpy(numpy_dtype)
        th.set_default_dtype(torch_dtype)

    else:
        raise ValueError(f"Unsupported dtype type: {type(dtype)}")

    return torch_dtype, numpy_dtype


def print_dtype_info():
    """Print current dtype configuration."""
    torch_dtype = th.get_default_dtype()
    numpy_dtype = get_numpy_dtype_from_torch(torch_dtype)

    print(f"Current PyTorch default dtype: {torch_dtype}")
    print(f"Corresponding NumPy dtype: {numpy_dtype}")
