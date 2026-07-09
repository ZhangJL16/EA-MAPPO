import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.UAVEnergyDeliveryLevel import (
    DeliveryOrder,
    TASK_CHARGE,
    UAVEnvDiscreteWrapper,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "greedy_threshold_charge",
            "energy_aware_greedy",
            "auction_threshold_charge",
        ],
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episode_limit", type=int, default=400)
    parser.add_argument("--uav_n_agents", type=int, default=4)
    parser.add_argument("--uav_total_orders", type=int, default=16)
    parser.add_argument("--uav_max_active_orders", type=int, default=8)
    parser.add_argument("--charge_threshold", type=float, default=0.35)
    parser.add_argument("--charge_release_threshold", type=float, default=0.65)
    parser.add_argument("--energy_reserve_ratio", type=float, default=0.04)
    parser.add_argument("--output_csv", default="")
    return parser.parse_args()


def order_target(order):
    if order.status == DeliveryOrder.PICKED:
        return order.dropoff_pos
    return order.pickup_pos


def finish_order_then_charge_distance(base, agent, order):
    current = agent.pos
    distance = 0.0
    if order.status == DeliveryOrder.ACTIVE:
        distance += float(np.linalg.norm(current - order.pickup_pos))
        distance += float(np.linalg.norm(order.pickup_pos - order.dropoff_pos))
    elif order.status == DeliveryOrder.ASSIGNED:
        distance += float(np.linalg.norm(current - order.pickup_pos))
        distance += float(np.linalg.norm(order.pickup_pos - order.dropoff_pos))
    elif order.status == DeliveryOrder.PICKED:
        distance += float(np.linalg.norm(current - order.dropoff_pos))
    distance += float(np.linalg.norm(order.dropoff_pos - base.charging_station_pos))
    return distance


def energy_feasible(base, agent, order, reserve_ratio):
    required_distance = finish_order_then_charge_distance(base, agent, order)
    required_steps = required_distance / max(agent.v_max * base.time_step, 1e-6)
    required_energy = required_steps * base.energy_decay_per_step
    reserve = reserve_ratio * agent.initial_energy
    return agent.energy >= required_energy + reserve


def available_order_slots(base):
    slots = []
    for slot_idx, order_id in enumerate(base.order_slots):
        if order_id is None or order_id >= len(base.orders):
            continue
        order = base.orders[order_id]
        if order.status == DeliveryOrder.ACTIVE and order.assigned_agent is None:
            slots.append(slot_idx)
    return slots


def choose_greedy_slot(base, agent, slots, energy_aware, reserve_ratio):
    best = None
    for slot_idx in slots:
        order = base._slot_order(slot_idx)
        if order is None:
            continue
        if energy_aware and not energy_feasible(base, agent, order, reserve_ratio):
            continue
        cost = float(np.linalg.norm(agent.pos - order.pickup_pos))
        if energy_aware:
            cost += 0.25 * finish_order_then_charge_distance(base, agent, order)
        if best is None or cost < best[0]:
            best = (cost, slot_idx)
    return None if best is None else int(best[1])


def update_assignments(env, method, charge_threshold, release_threshold, reserve_ratio):
    base = env.env
    base.charge_release_threshold = float(release_threshold)
    base._activate_orders()

    if method == "auction_threshold_charge":
        high_actions = []
        for agent in base.agents:
            ratio = agent.energy / max(agent.initial_energy, 1e-6)
            mode = 0.0 if ratio <= charge_threshold else 1.0
            high_actions.append(np.array([mode, 1.0], dtype=np.float32))
        env.apply_high_level_actions(high_actions)
        return

    claimed = set()
    for agent in base.agents:
        if not agent.has_energy():
            continue
        if agent.assigned_order_slot is not None:
            base._assign_order_slot_to_agent(agent, agent.assigned_order_slot)
            continue
        if agent.current_task_type == TASK_CHARGE:
            if base._charge_option_complete(agent):
                base._set_agent_idle(agent)
            else:
                base._set_agent_charging(agent)
                continue

        energy_ratio = agent.energy / max(agent.initial_energy, 1e-6)
        if energy_ratio <= charge_threshold:
            base._set_agent_charging(agent)
            continue

        slots = [slot for slot in available_order_slots(base) if slot not in claimed]
        slot_idx = choose_greedy_slot(
            base,
            agent,
            slots,
            energy_aware=(method == "energy_aware_greedy"),
            reserve_ratio=reserve_ratio,
        )
        if slot_idx is None:
            continue
        if base._assign_order_slot_to_agent(agent, slot_idx):
            claimed.add(slot_idx)


def choose_low_action(env, agent_id):
    base = env.env
    agent = base.agents[agent_id]
    if not agent.has_energy() or not base._agent_has_motion_task(agent):
        return env.get_noop_action()
    direction = agent.goal - agent.pos
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return env.get_noop_action()
    direction = direction / norm
    scores = [
        float(np.dot(action, direction)) - 0.01 * float(np.linalg.norm(action))
        for action in env.discrete_actions
    ]
    return int(np.argmax(scores))


def run_episode(args, episode_idx):
    env = UAVEnvDiscreteWrapper(
        dim_actions=2,
        num_hunters=args.uav_n_agents,
        episode_limit=args.episode_limit,
        total_orders=args.uav_total_orders,
        max_active_orders=args.uav_max_active_orders,
    )
    env.reset(seed=args.seed + episode_idx)
    total_reward = 0.0
    terminated = False
    steps = 0
    while not terminated and steps < args.episode_limit:
        update_assignments(
            env,
            args.method,
            args.charge_threshold,
            args.charge_release_threshold,
            args.energy_reserve_ratio,
        )
        actions = [choose_low_action(env, agent_id) for agent_id in range(env.n_agents)]
        reward, terminated, info = env.step(actions)
        total_reward += float(reward)
        steps += 1

    summary = env.summary()
    row = {
        "method": args.method,
        "episode": episode_idx,
        "episode_reward": total_reward,
        "episode_steps": steps,
        "orders_completed": summary.get("orders_completed", 0.0),
        "total_orders": summary.get("total_orders", 0.0),
        "collision_count": summary.get("collision_count", 0.0),
        "obstacle_collision_count": summary.get("obstacle_collision_count", 0.0),
        "agent_collision_count": summary.get("agent_collision_count", 0.0),
        "charging_agents": summary.get("charging_agents", 0.0),
        "mean_energy": summary.get("mean_energy", 0.0),
        "depleted_agents": summary.get("depleted_agents", 0.0),
        "win_tag": summary.get("win_tag", False),
    }
    env.close()
    return row


def main():
    args = parse_args()
    rows = [run_episode(args, episode_idx) for episode_idx in range(args.episodes)]
    output_csv = args.output_csv
    if not output_csv:
        output_dir = Path("logs/uav_energy_delivery_comparisons") / args.method
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = str(output_dir / "heuristic_eval.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    means = {
        key: np.mean([float(row[key]) for row in rows])
        for key in (
            "episode_reward",
            "orders_completed",
            "collision_count",
            "obstacle_collision_count",
            "agent_collision_count",
            "mean_energy",
        )
    }
    print(f"Saved heuristic results to {output_csv}")
    print(means)


if __name__ == "__main__":
    main()
