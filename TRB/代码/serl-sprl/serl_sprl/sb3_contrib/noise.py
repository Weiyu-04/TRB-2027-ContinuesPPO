import numpy as np
from numpy.typing import DTypeLike
from stable_baselines3.common.noise import ActionNoise


class LinearNormalActionNoise(ActionNoise):
    """
    A Gaussian action noise that linearly decreases over time.

    :param mean: Mean value of the noise
    :param sigma: Scale of the noise (std here)
    :param dtype: Type of the output noise
    :param max_steps: Total  number of steps for the noise to decay
    :param final_sigma: Final value of the noise (std here)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        max_steps: int,
        final_sigma: np.ndarray = None,
        dtype: DTypeLike = np.float32,
    ) -> None:
        self._mu = mean
        self._sigma = sigma
        self._max_steps = max_steps
        if final_sigma is None:
            final_sigma = np.zeros_like(sigma)
        self._final_sigma = final_sigma
        self._step = 0
        self._dtype = dtype
        super().__init__()

    def __call__(self) -> np.ndarray:
        t = min(1.0, self._step / self._max_steps)
        sigma = self._sigma * (1 - t) + self._final_sigma * t
        self._step += 1
        return np.random.normal(self._mu, sigma).astype(self._dtype)

    def __repr__(self) -> str:
        return f"LinearNormalActionNoise(mu={self._mu}, sigma={self._sigma})"
