#!/usr/bin/env python3
"""Generate a GIF rollout for the current UAVEnergyDeliveryLevel H-MAPPO model."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.agent import Agents
from common.arguments import get_common_args, get_mappo_args
from common.rollout import _get_active_agent_mask, _get_env_msg, _get_noop_action
from common.seeding import reset_env_with_seed, seed_everything
from main_level import build_env


DEFAULT_MODEL_DIR = (
    "model_runs/twostage/"
    "uedl_hmappo_twostage_0709_231113_DESKTOP-5C28MHK__"
    "hmappo_twostage_lowpretrain_seed123_high"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval_seed", type=int, default=100123)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--duration_ms", type=int, default=90)
    parser.add_argument("--cuda", default="True")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--view", choices=["xy", "3d"], default="xy")
    return parser.parse_args()


def build_hmappo_args(script_args):
    saved_argv = sys.argv[:]
    sys.argv = [
        "main_level.py",
        "--alg",
        "hmappo",
        "--map",
        "UAVEnergyDeliveryLevel",
        "--uav_n_agents",
        "4",
        "--episode_limit",
        str(script_args.max_steps),
        "--uav_total_orders",
        "16",
        "--uav_max_active_orders",
        "8",
        "--hmappo_meta_period",
        "5",
        "--hmappo_pretrained_low_model_dir",
        "",
        "--hmappo_freeze_low_level",
        "False",
        "--hmappo_freeze_high_level",
        "False",
        "--hrl_meta_update_on_subgoal_done",
        "False",
        "--hrl_order_progress_override",
        "0.65",
        "--hrl_delivery_intrinsic_progress_bonus",
        "0.0",
        "--hrl_intrinsic_collision_penalty",
        "0.8",
        "--hrl_charge_dense_reward_scale",
        "0.0",
        "--hrl_energy_shield_enabled",
        "True",
        "--hrl_energy_margin_reserve_ratio",
        "0.08",
        "--hrl_energy_margin_loss_coef",
        "0.2",
        "--hrl_energy_margin_charge_beta",
        "0.2",
        "--hrl_charge_queue_enabled",
        "True",
        "--hrl_charge_queue_radius",
        "0.24",
        "--hrl_safe_action_guard_enabled",
        "True",
        "--hrl_safe_action_guard_margin",
        "0.04",
        "--load_model",
        "True",
        "--model_dir",
        script_args.model_dir,
        "--replay_dir",
        "",
        "--seed",
        str(script_args.seed),
        "--eval_seed",
        str(script_args.eval_seed),
        "--cuda",
        str(script_args.cuda),
        "--gpu_id",
        str(script_args.gpu_id),
    ]
    try:
        args = get_common_args()
    finally:
        sys.argv = saved_argv

    args.use_level_policy = True
    args.is_level_training = True
    args.now = "gif"
    args.evaluate = True
    return args


def configure_env_and_args(args):
    seed_everything(
        args.seed,
        deterministic_torch=bool(getattr(args, "deterministic_torch", True)),
    )
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(args, "gpu_id", 0)))

    env = build_env(args, [args.alg])
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.raw_obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.msg_shape = env_info.get("msg_shape", 0)
    args.high_level_n_actions = env_info.get("high_level_n_actions", 0)
    args.high_level_mode_n_actions = env_info.get("high_level_mode_n_actions", 0)
    args.high_level_obs_shape = env_info.get("high_level_obs_shape", 0)
    args.high_level_state_shape = env_info.get("high_level_state_shape", 0)
    args.low_task_shape = env_info.get("low_task_shape", 0)
    args.max_active_orders = env_info.get(
        "max_active_orders", getattr(args, "uav_max_active_orders", 0)
    )
    args.charge_action_id = env_info.get("charge_action_id", args.max_active_orders)
    args = get_mappo_args(args)
    return env, args


def maybe_set_hrl_parameters(env, args):
    if hasattr(env, "set_meta_period"):
        env.set_meta_period(max(1, int(getattr(args, "hmappo_meta_period", 5))))
    if hasattr(env, "set_hrl_parameters"):
        env.set_hrl_parameters(
            reachable_subgoal_scale=getattr(args, "hrl_reachable_subgoal_scale", None),
            intrinsic_reward_scale=getattr(args, "hrl_intrinsic_reward_scale", None),
            intrinsic_distance_weight=getattr(args, "hrl_intrinsic_distance_weight", None),
            intrinsic_success_bonus=getattr(args, "hrl_intrinsic_success_bonus", None),
            delivery_intrinsic_progress_bonus=getattr(
                args, "hrl_delivery_intrinsic_progress_bonus", None
            ),
            intrinsic_collision_penalty=getattr(args, "hrl_intrinsic_collision_penalty", None),
            order_progress_override=getattr(args, "hrl_order_progress_override", None),
            energy_shield_enabled=getattr(args, "hrl_energy_shield_enabled", None),
            energy_margin_reserve_ratio=getattr(args, "hrl_energy_margin_reserve_ratio", None),
            charge_queue_enabled=getattr(args, "hrl_charge_queue_enabled", None),
            charge_queue_radius=getattr(args, "hrl_charge_queue_radius", None),
        )


def render_frame(env, view):
    render_env = env
    if not hasattr(render_env, "render") and hasattr(render_env, "env"):
        render_env = render_env.env
    frame = render_env.render(show=False, view=None if view == "3d" else "xy")
    if frame is None:
        raise RuntimeError("Environment render returned None.")
    return Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3])


@torch.no_grad()
def rollout_to_gif(env, agents, args, output, frame_stride, duration_ms, view):
    reset_env_with_seed(env, args.eval_seed)
    if hasattr(agents, "reset_episode_state"):
        agents.reset_episode_state()
    maybe_set_hrl_parameters(env, args)
    agents.policy.init_hidden(1)

    frames = [render_frame(env, view)]
    last_action = np.zeros((args.n_agents, args.n_actions), dtype=np.float32)
    meta_period = max(1, int(getattr(args, "hmappo_meta_period", 5)))
    terminated = False
    episode_reward = 0.0
    step = 0
    last_info = {}

    while not terminated and step < args.episode_limit:
        active_agent_mask = _get_active_agent_mask(env, args.n_agents)
        noop_action = _get_noop_action(env, args.n_actions)

        if step % meta_period == 0:
            if hasattr(env, "prepare_high_level_decision"):
                env.prepare_high_level_decision()
            high_obs = env.get_high_level_obs()
            high_action_dim = int(getattr(args, "high_level_n_actions", 0))
            high_avail = np.ones((args.n_agents, high_action_dim), dtype=np.float32)
            high_actions = []
            for agent_id in range(args.n_agents):
                if active_agent_mask[agent_id] <= 0.0:
                    high_action = np.zeros(high_action_dim, dtype=np.float32)
                else:
                    high_action = agents.choose_high_level_action(
                        high_obs[agent_id],
                        agent_id,
                        high_avail[agent_id],
                        epsilon=0.0,
                    )
                high_actions.append(np.asarray(high_action, dtype=np.float32))
            env.apply_high_level_actions(high_actions)

        obs = env.get_obs()
        raw_obs = np.asarray(obs, dtype=np.float32).copy()
        msg = _get_env_msg(env, args, args.n_agents)
        actions = []
        avail_actions = []
        for agent_id in range(args.n_agents):
            avail_action = env.get_avail_agent_actions(agent_id)
            if active_agent_mask[agent_id] <= 0.0:
                action = noop_action
            else:
                action = agents.choose_action(
                    obs[agent_id],
                    last_action[agent_id],
                    agent_id,
                    avail_action,
                    epsilon=0.0,
                    timestep_cur=step,
                    timestep_max=args.n_steps,
                    msg=None if msg is None else msg[agent_id],
                )
            action = int(action)
            actions.append(action)
            avail_actions.append(avail_action)

        if hasattr(agents, "revise_safe_actions"):
            revised_actions = agents.revise_safe_actions(
                observations=raw_obs,
                avail_actions=avail_actions,
                base_actions=actions,
            )
            if revised_actions is not None:
                actions = [
                    int(action) if active_agent_mask[idx] > 0.0 else noop_action
                    for idx, action in enumerate(revised_actions)
                ]

        for agent_id, action in enumerate(actions):
            last_action[agent_id] = 0.0
            last_action[agent_id, int(action)] = 1.0

        reward, terminated, last_info = env.step(actions)
        episode_reward += float(np.asarray(reward, dtype=np.float32).mean())
        step += 1
        if step % frame_stride == 0 or terminated:
            frames.append(render_frame(env, view))

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    summary = env.summary() if hasattr(env, "summary") else {}
    return {
        "output": str(output),
        "frames": len(frames),
        "steps": step,
        "episode_reward": episode_reward,
        "terminated": bool(terminated),
        "orders_completed": summary.get("orders_completed"),
        "collision_count": summary.get("collision_count"),
        "obstacle_collision_count": summary.get("obstacle_collision_count"),
        "agent_collision_count": summary.get("agent_collision_count"),
        "info": last_info,
    }


def main():
    script_args = parse_args()
    model_dir = Path(script_args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    args = build_hmappo_args(script_args)
    env, args = configure_env_and_args(args)
    agents = Agents(args, env)

    if script_args.output:
        output = Path(script_args.output)
    else:
        output = Path("replay_gifs") / f"current_hmappo_seed{script_args.seed}_eval{script_args.eval_seed}.gif"

    try:
        summary = rollout_to_gif(
            env,
            agents,
            args,
            output=output,
            frame_stride=max(1, int(script_args.frame_stride)),
            duration_ms=max(20, int(script_args.duration_ms)),
            view=script_args.view,
        )
    finally:
        if hasattr(env, "close"):
            env.close()

    for key, value in summary.items():
        if key != "info":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
