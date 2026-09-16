from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

from .obstacles import AABBObstacle, CylinderObstacle, StaticWorld
from .state import NavigationState, as_vec3


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    world_size: np.ndarray
    initial_state: NavigationState
    station_position: np.ndarray
    task_goal: np.ndarray
    world: StaticWorld
    configuration_overrides: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_size", as_vec3(self.world_size, "world_size"))
        object.__setattr__(self, "station_position", as_vec3(self.station_position, "station_position"))
        object.__setattr__(self, "task_goal", as_vec3(self.task_goal, "task_goal"))


def load_scenario(path_or_name: str | Path) -> ScenarioDefinition:
    path = Path(path_or_name)
    if not path.exists():
        path = Path(str(files("envs.navigation.scenarios").joinpath(str(path_or_name))))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    world_size = np.asarray(payload["world_size"], dtype=np.float64)
    aabbs = tuple(
        AABBObstacle(np.asarray(item["low"], dtype=np.float64), np.asarray(item["high"], dtype=np.float64))
        for item in payload.get("aabb_obstacles", ())
    )
    cylinders = tuple(
        CylinderObstacle(
            np.asarray(item["center_xy"], dtype=np.float64),
            float(item["radius"]),
            float(item.get("z_low", 0.0)),
            float(item.get("z_high", world_size[2])),
        )
        for item in payload.get("cylinder_obstacles", ())
    )
    initial = payload["initial_state"]
    return ScenarioDefinition(
        name=str(payload["name"]),
        world_size=world_size,
        initial_state=NavigationState(
            np.asarray(initial["position"], dtype=np.float64),
            np.asarray(initial["velocity"], dtype=np.float64),
            float(initial["energy"]),
            float(initial.get("timestamp", 0.0)),
        ),
        station_position=np.asarray(payload["station_position"], dtype=np.float64),
        task_goal=np.asarray(payload["task_goal"], dtype=np.float64),
        world=StaticWorld(world_size, aabbs, cylinders),
        configuration_overrides=dict(payload.get("configuration_overrides", {})),
    )


class NavigationScenario:
    def __init__(self, name: str = "random_persistent_open.json") -> None:
        self.definition = load_scenario(name)
