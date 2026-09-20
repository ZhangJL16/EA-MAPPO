"""Reset-state census of the unmodified SAC environment; no rollout or training.

Run from repository root:
PYTHONPATH=.:runtime_support .venv/bin/python research/full_delivery_audit/audit_structure.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv


def main():
    root = Path(__file__).resolve().parents[2]
    sources = [root / 'envs/UAVEnergyDeliverySAC.py',
               root / 'runtime_support/review_bundle/envs/navigation/telemetry_cost.py']
    rows = []
    env = UAVEnergyDeliverySACEnv()
    try:
        for seed in range(20260920, 20260936):
            obs, _ = env.reset(seed=seed)
            assert np.isfinite(obs).all()
            assert env.orders == [] and env.current_task_point.shape == (3,)
            distance = float(np.linalg.norm(env.current_task_point - env.agent.pos))
            rows.append(dict(
                state_id=f'reset-{seed}', seed=seed, time=env.simulation_time,
                position=env.agent.pos.tolist(), battery=float(env.agent.energy),
                finite_energy_enabled=env.finite_energy_enabled,
                candidate_tasks=[dict(kind='single_navigation_goal',
                                      target=env.current_task_point.tolist(),
                                      pickup=None, dropoff=None, deadline=None, payload=None,
                                      euclidean_distance=distance,
                                      estimated_duration=None, estimated_energy=None)],
                current_load=None, legacy_order_count=len(env.orders),
                charger_state=dict(position=env.charger_position.tolist(), count=1),
                action_taken=None, successor=None, task_completed=False,
            ))
    finally:
        env.close()
    distances = [r['candidate_tasks'][0]['euclidean_distance'] for r in rows]
    report = dict(
        scope='F0 structural verification and reset-state census only; NOT F1 trajectory statistics or F2 performance',
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        config='Unmodified constructor defaults: NAVIGATION, no obstacles, no finite battery; battery=1 is a placeholder.',
        interpretation='One target is structurally available; feasible task count is NOT measured. Null estimates/outcomes are unavailable, not zero.',
        training_updates=0, policy_steps=0, reset_states=rows,
        summary=dict(n=len(rows), target_count_distribution={'1': len(rows)},
                     legacy_order_count_distribution={'0': len(rows)},
                     distance_min=min(distances), distance_mean=float(np.mean(distances)),
                     distance_max=max(distances)),
    )
    output = Path(__file__).with_name('reset_state_census.json')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
