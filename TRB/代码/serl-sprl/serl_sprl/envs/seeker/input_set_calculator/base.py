from typing import Any

from continuoussets.convexsets.convexset import ConvexSet
from gymnasium.core import ActType, ObsType


class RelevantInputSetCalculator:
    def compute_input_set(self, observation: ObsType, info: dict[str, Any]) -> ConvexSet:
        raise NotImplementedError


class SafeInputCalculator:
    def compute_input(self, observation: ObsType, info: dict[str, Any], action: ActType | None) -> ActType:
        raise NotImplementedError
