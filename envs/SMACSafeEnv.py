import numpy as np


class SMACSafetyWrapper:
    """Add a paper-style safety penalty signal for SMAC environments.

    The penalty follows the SMAC description in the paper: remaining life,
    shield value, and agent-enemy distance are used to evaluate action risk.

    This wrapper also exposes:
    - `warning_signal`/`safety_penalty` in `step()` info;
    - `estimate_joint_short_risk(actions)`;
    - `estimate_joint_short_risk_batch(actions_batch)`;
    - `estimate_joint_collision_flags(actions)` used by the safety guide as a hard gate.
    """

    def __init__(
        self,
        base_env,
        risk_threshold=0.1,
        life_weight=0.4,
        shield_weight=0.2,
        distance_weight=0.4,
    ):
        self.base_env = base_env
        self.risk_threshold = float(risk_threshold)
        self.life_weight = float(life_weight)
        self.shield_weight = float(shield_weight)
        self.distance_weight = float(distance_weight)
        self.warning_signal = np.zeros(self.n_agents, dtype=np.float32)
        self._last_info = {}
        self._max_life = np.ones(self.n_agents, dtype=np.float32)
        self._max_shield = np.zeros(self.n_agents, dtype=np.float32)

    def __getattr__(self, item):
        return getattr(self.base_env, item)

    @property
    def n_agents(self):
        return int(getattr(self.base_env, "n_agents"))

    @property
    def n_actions(self):
        return int(getattr(self.base_env, "n_actions"))

    def reset(self):
        obs = self.base_env.reset()
        self._refresh_reference_stats()
        self.warning_signal = np.zeros(self.n_agents, dtype=np.float32)
        self._last_info = {
            "battle_won": False,
            "warning_signal": self.warning_signal.reshape(self.n_agents, 1).copy(),
        }
        return obs

    def step(self, actions):
        reward, terminated, info = self.base_env.step(actions)
        penalty = self._safety_penalty(actions=None)
        self.warning_signal = penalty.astype(np.float32)
        wrapped_info = dict(info) if isinstance(info, dict) else {}
        wrapped_info["warning_signal"] = self.warning_signal.reshape(self.n_agents, 1)
        wrapped_info["safety_penalty"] = self.warning_signal.reshape(self.n_agents, 1)
        self._last_info = wrapped_info
        return reward, terminated, wrapped_info

    def estimate_joint_short_risk(self, actions):
        return self._safety_penalty(actions).astype(np.float32)

    def estimate_joint_short_risk_batch(self, actions_batch):
        return np.asarray(
            [self._safety_penalty(actions) for actions in actions_batch],
            dtype=np.float32,
        )

    def estimate_joint_collision_flags(self, actions):
        risk = self._safety_penalty(actions)
        return (risk > self.risk_threshold).astype(np.float32)

    def _refresh_reference_stats(self):
        ally_units = list(getattr(self.base_env, "agents", []))
        if not ally_units:
            return
        for agent_idx, ally in enumerate(ally_units[: self.n_agents]):
            self._max_life[agent_idx] = max(self._unit_life(ally), 1.0)
            self._max_shield[agent_idx] = max(self._unit_shield(ally), 0.0)

    def _safety_penalty(self, actions=None):
        ally_units = list(getattr(self.base_env, "agents", []))
        enemy_units = list(getattr(self.base_env, "enemies", []))
        if not ally_units:
            return np.zeros(self.n_agents, dtype=np.float32)

        predicted_positions = self._predict_ally_positions(ally_units, actions)
        penalties = np.zeros(self.n_agents, dtype=np.float32)

        for agent_idx, ally in enumerate(ally_units):
            if not self._unit_alive(ally):
                continue

            life_risk = self._life_risk(agent_idx, ally)
            shield_risk = self._shield_risk(agent_idx, ally)
            distance_risk = self._distance_risk(predicted_positions[agent_idx], enemy_units)
            penalty = (
                self.life_weight * life_risk
                + self.shield_weight * shield_risk
                + self.distance_weight * distance_risk
            )
            penalties[agent_idx] = float(np.clip(penalty, 0.0, 1.0))

        return penalties

    def _life_risk(self, agent_idx, ally):
        max_life = max(float(self._max_life[agent_idx]), self._unit_life(ally), 1e-6)
        return float(np.clip(1.0 - self._unit_life(ally) / max_life, 0.0, 1.0))

    def _shield_risk(self, agent_idx, ally):
        max_shield = max(float(self._max_shield[agent_idx]), self._unit_shield(ally))
        if max_shield <= 1e-6:
            return 0.0
        return float(np.clip(1.0 - self._unit_shield(ally) / max_shield, 0.0, 1.0))

    def _distance_risk(self, ally_pos, enemy_units):
        max_risk = 0.0
        for enemy in enemy_units:
            if not self._unit_alive(enemy):
                continue
            enemy_pos = self._unit_xy(enemy)
            distance = float(np.linalg.norm(ally_pos - enemy_pos))
            risk_radius = max(self._unit_attack_range(enemy) * 1.5, 1.0)
            risk = np.clip(1.0 - distance / risk_radius, 0.0, 1.0)
            max_risk = max(max_risk, float(risk))
        return max_risk

    def _predict_ally_positions(self, ally_units, actions):
        predicted = np.asarray([self._unit_xy(unit) for unit in ally_units], dtype=np.float32)
        if actions is None:
            return predicted

        move_amount = float(
            getattr(self.base_env, "move_amount", getattr(self.base_env, "_move_amount", 2.0))
        )
        deltas = {
            2: np.array([0.0, move_amount], dtype=np.float32),   # north
            3: np.array([0.0, -move_amount], dtype=np.float32),  # south
            4: np.array([move_amount, 0.0], dtype=np.float32),   # east
            5: np.array([-move_amount, 0.0], dtype=np.float32),  # west
        }
        for idx, action in enumerate(actions[: len(ally_units)]):
            delta = deltas.get(int(action))
            if delta is not None:
                predicted[idx] = predicted[idx] + delta
        return predicted

    def _unit_xy(self, unit):
        if hasattr(unit, "pos") and unit.pos is not None:
            x = getattr(unit.pos, "x", None)
            y = getattr(unit.pos, "y", None)
            if x is not None and y is not None:
                return np.asarray([float(x), float(y)], dtype=np.float32)
        return np.asarray(
            [
                float(getattr(unit, "x", 0.0)),
                float(getattr(unit, "y", 0.0)),
            ],
            dtype=np.float32,
        )

    def _unit_life(self, unit):
        return float(max(getattr(unit, "health", 0.0), 0.0))

    def _unit_shield(self, unit):
        return float(max(getattr(unit, "shield", 0.0), 0.0))

    def _unit_health(self, unit):
        return self._unit_life(unit) + self._unit_shield(unit)

    def _unit_alive(self, unit):
        return self._unit_health(unit) > 0.0

    def _unit_attack_range(self, unit):
        for attr in ("shoot_range", "attack_range", "range"):
            value = getattr(unit, attr, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return 6.0
