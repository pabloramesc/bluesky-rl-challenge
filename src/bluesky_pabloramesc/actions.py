"""Explicit units and mappings for heading/speed ablations."""

import math
from dataclasses import asdict, dataclass

import numpy as np

KNOTS_PER_MPS = 1.94384


@dataclass(frozen=True)
class ControlConfig:
    """Policy interface. Speed bounds and deltas are in knots."""

    heading_action: str = "relative"
    speed_action: str = "relative"
    observation_profile: str = "legacy"
    speed_min: float | None = None
    speed_max: float | None = None
    heading_delta: float = 45.0
    speed_delta: float = 20 / 3
    schema_version: int = 1
    observation_encoding: str = "upstream"
    decision_interval: int = 10

    def __post_init__(self):
        if type(self.decision_interval) is not int or self.decision_interval <= 0:
            raise ValueError("decision interval must be a positive integer")
        for value in (self.heading_action, self.speed_action):
            if value not in ("relative", "absolute"):
                raise ValueError("action mode must be relative or absolute")
        if self.observation_profile not in ("legacy", "control"):
            raise ValueError("observation profile must be legacy or control")
        if self.observation_encoding not in (
            "upstream",
            "ego_cartesian",
            "goal_aligned",
            "conflict",
            "ego_cartesian_conflict",
            "goal_aligned_conflict",
        ):
            raise ValueError("unknown observation encoding")
        if self.heading_action == "absolute" and self.observation_profile != "control":
            raise ValueError("absolute heading requires --observation-profile control")
        if (self.speed_min is None) != (self.speed_max is None):
            raise ValueError("supply both --speed-min and --speed-max (CAS knots)")
        if self.speed_min is not None and not (
            math.isfinite(self.speed_min)
            and math.isfinite(self.speed_max)
            and 0 < self.speed_min < self.speed_max
        ):
            raise ValueError("speed bounds must be finite with 0 < min < max")
        if self.speed_action == "absolute" and self.speed_min is None:
            raise ValueError("absolute speed requires --speed-min and --speed-max")
        if not all(
            math.isfinite(x) and x > 0 for x in (self.heading_delta, self.speed_delta)
        ):
            raise ValueError("action deltas must be finite and positive")
        if self.schema_version != 1:
            raise ValueError("unsupported control schema version")

    def to_dict(self):
        """Return JSON-serializable settings for checkpoint metadata."""
        return asdict(self)


def action_targets(action, heading, cas_knots, config):
    """Map normalized actions to heading degrees and CAS knots."""
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("expected two finite heading/speed actions")
    turn, speed = np.clip(action, -1, 1)
    target_heading = (
        heading + turn * config.heading_delta
        if config.heading_action == "relative"
        else 180 * (turn + 1)
    )
    target_speed = (
        cas_knots + speed * config.speed_delta
        if config.speed_action == "relative"
        else config.speed_min + (speed + 1) / 2 * (config.speed_max - config.speed_min)
    )
    if config.speed_min is not None:
        target_speed = np.clip(target_speed, config.speed_min, config.speed_max)
    return (target_heading + 180) % 360 - 180, float(target_speed)
