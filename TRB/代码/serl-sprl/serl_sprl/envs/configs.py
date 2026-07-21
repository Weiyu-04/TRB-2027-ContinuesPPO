from typing import Any, Callable, Optional

import numpy as np
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class BaseEnvConfig(BaseModel):
    id: str
    max_rollout_steps: int
    dtype: type = np.float32
    multi_step_safeguarding: bool = False


class BaseProjectionConfig(BaseModel):
    scale_actions: bool = True
    safe_control_fn: Optional[Callable] = None
    safe_set_calculator: Optional[Any] = None
    penalty_factor: float = 0.0
