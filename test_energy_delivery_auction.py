import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from agent.agent import Agents
from main import build_env
from main_energy_delivery_auction import (
    apply_env_info,
    configure_algorithm_args,
    install_auction_assignment,
)


def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return int_value


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate UAVEnergyDelivery MAPPO with optional auction assignment."
    )
    parser.add_argument("--assignment", choices=["nearest", "auction"], default="auction")
    parser.add_argument("--alg", default="mappo")
    parser.add_argument("--map", default="UAVEnergyDelivery")
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--episode-limit", "--episode_limit", dest="episode_limit", type=positive_int, default=400)
    parser.add_argument("--uav-n-agents", "--uav_n_agents", dest="uav_n_agents", type=positive_int, default=4)
    parser.add_argument("--uav-total-orders", "--uav_total_orders", dest="uav_total_orders", type=positive_int, default=16)
    parser.add_argument("--uav-max-active-orders", "--uav_max_active_orders", dest="uav_max_active_orders", type=positive_int, default=8)
    parser.add_argument("--uav-pickup-reward", "--uav_pickup_reward", dest="uav_pickup_reward", type=float, default=3.0)
    parser.add_argument("--uav-delivery-reward", "--uav_delivery_reward", dest="uav_delivery_reward", type=float, default=8.0)
    parser.add_argument("--uav-initial-energy", "--uav_initial_energy", dest="uav_initial_energy", type=float, default=100.0)
    parser.add_argument("--uav-energy-decay", "--uav_energy_decay", dest="uav_energy_decay", type=float, default=None)
    parser.add_argument("--uav-energy-depletion-fraction", "--uav_energy_depletion_fraction", dest="uav_energy_depletion_fraction", type=float, default=0.5)
    parser.add_argument("--uav-charging-capacity", "--uav_charging_capacity", dest="uav_charging_capacity", type=positive_int, default=2)
    parser.add_argument("--uav-charging-radius", "--uav_charging_radius", dest="uav_charging_radius", type=float, default=0.18)
    parser.add_argument("--uav-charging-rate", "--uav_charging_rate", dest="uav_charging_rate", type=float, default=None)
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir", default="./model")
    parser.add_argument("--result-root", default="./result/test_energy_delivery_auction")
    parser.add_argument("--render-mode", choices=["none", "rgb_array"], default="none")
    parser.add_argument("--gif-fps", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=0)
    parser.add_argument("--load-model", "--load_model", dest="load_model", type=str2bool, default=True)
    return parser.parse_args()


def make_train_like_args(cli_args):
    args = SimpleNamespace(**vars(cli_args))
    args.alg_list = ""
    args.replay_dir = ""
    args.debug = False
    args.sc2_path = ""
    args.difficulty = "7"
    args.game_version = "latest"
    args.step_mul = 8
    args.last_action = False
    args.reuse_network = True
    args.gamma = 0.99
    args.optimizer = "RMS"
    args.result_dir = "./result"
    args.evaluate = True
    args.now = "auction_test"
    args.n_steps = 1
    args.time_steps = None
    args.n_episodes = 1
    args.evaluate_cycle = 1
    args.evaluate_epoch = 1
    args.smac_parallel_envs = 1
    args.deterministic_torch = True
    args.eval_seed = args.seed + 100000
    args.episode_seed_stride = 1
    args.comm = False
    args.distributed = False
    args.model_dir = cli_args.model_dir
    args.load_model = cli_args.load_model
    args.high_lr_actor = None
    args.high_lr_critic = None
    args.high_actor_hidden_dim = None
    args.high_critic_hidden_dim = None
    args.hrl_off_policy_correction = False
    args.safety_lr = None
    args.safety_beta = None
    args.comm_lr = None
    args.comm_max_keep_dim = 8
    args.run_script = ""
    args.run_command = ""
    return args


def save_gif(frames, output_path, fps):
    if not frames:
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    duration_ms = int(round(1000 / max(1, fps)))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def run_episode(env, agents, args, episode_idx):
    seed = args.seed + episode_idx * args.episode_seed_stride
    env.reset(seed=seed)
    last_action = np.zeros((args.n_agents, args.n_actions), dtype=np.float32)
    episode_reward = 0.0
    frames = []

    for step in range(args.episode_limit):
        obs = env.get_obs()
        actions = []
        for agent_id in range(args.n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            action = agents.choose_action(
                obs[agent_id],
                last_action[agent_id],
                agent_id,
                avail_actions,
                epsilon=0.0,
            )
            actions.append(int(action))
            last_action[agent_id] = 0.0
            last_action[agent_id, int(action)] = 1.0
        reward, terminated, _ = env.step(actions)
        episode_reward += float(reward)
        if args.render_mode == "rgb_array":
            frames.append(env.render(view="rgb_array"))
        if terminated:
            break

    summary = env.summary()
    summary["episode_reward_sum"] = episode_reward
    summary["episode_steps"] = float(step + 1)
    return summary, frames


def main():
    cli_args = parse_args()
    args = make_train_like_args(cli_args)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(args, "gpu_id", 0)))

    env = build_env(args, [args.alg])
    if args.assignment == "auction":
        install_auction_assignment(env)
    args = apply_env_info(args, env)
    args = configure_algorithm_args(args)

    agents = Agents(args, env)
    summaries = []
    saved_gifs = []
    for episode_idx in range(cli_args.episodes):
        summary, frames = run_episode(env, agents, args, episode_idx)
        summaries.append(summary)
        if cli_args.render_mode == "rgb_array":
            gif_path = os.path.join(
                cli_args.result_root,
                cli_args.assignment,
                f"episode_{episode_idx}.gif",
            )
            saved = save_gif(frames, gif_path, cli_args.gif_fps)
            if saved:
                saved_gifs.append(saved)

    keys = [
        "episode_reward_sum",
        "orders_completed",
        "picked_orders",
        "active_orders",
        "available_orders",
        "charging_agents",
        "mean_energy",
        "collision_count",
        "obstacle_collision_count",
        "agent_collision_count",
        "episode_steps",
    ]
    print(f"assignment={cli_args.assignment}")
    for key in keys:
        values = [float(summary.get(key, 0.0)) for summary in summaries]
        print(f"{key}: mean={np.mean(values):.3f}, values={[round(v, 3) for v in values]}")
    for path in saved_gifs:
        print(f"saved_gif: {path}")
    env.close()


if __name__ == "__main__":
    main()
