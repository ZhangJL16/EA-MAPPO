import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from agent.agent import Agents
from common.arguments import (
    get_centralv_args,
    get_coma_args,
    get_commnet_args,
    get_g2anet_args,
    get_macpo_args,
    get_mappo_args,
    get_mixer_args,
    get_reinforce_args,
)
from common.rollout import _build_env_summary, _get_env_msg
from main import build_env


def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return int_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and run evaluation episodes."
    )
    parser.add_argument("--alg", default="mappo", help="Algorithm name")
    parser.add_argument("--map", default="UAV3D", help="Environment map name")
    parser.add_argument(
        "--episodes", type=positive_int, default=1, help="Number of test episodes"
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional per-episode step cap. Defaults to env episode_limit.",
    )
    parser.add_argument(
        "--model-dir", default="./model", help="Root directory of saved checkpoints"
    )
    parser.add_argument(
        "--result-root",
        default="./result/test",
        help="Directory for saved evaluation GIFs",
    )
    parser.add_argument(
        "--frame-root",
        default="./test_result",
        help="Directory for saved per-frame 3D and XY-view images",
    )
    parser.add_argument(
        "--render-mode",
        default="rgb_array",
        choices=["none", "rgb_array", "human"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable visualization",
    )
    parser.add_argument(
        "--gif-fps",
        type=positive_int,
        default=8,
        help="Playback FPS for saved GIFs",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=20,
        help="Refresh rate for human rendering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed. Only used if the environment reset supports it.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if the loaded policy supports it",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id to use when CUDA is enabled",
    )
    parser.add_argument(
        "--uav-n-agents",
        "--uav_n_agents",
        dest="uav_n_agents",
        type=positive_int,
        default=4,
        help="Number of UAV agents for UAV2D/UAV3D environments",
    )
    parser.add_argument(
        "--episode-limit",
        "--episode_limit",
        dest="episode_limit",
        type=positive_int,
        default=400,
        help="Episode length for local UAV environments",
    )
    parser.add_argument(
        "--uav-total-orders",
        "--uav_total_orders",
        dest="uav_total_orders",
        type=positive_int,
        default=16,
        help="Total delivery orders per UAVDelivery/UAVEnergyDelivery episode",
    )
    parser.add_argument(
        "--uav-max-active-orders",
        "--uav_max_active_orders",
        dest="uav_max_active_orders",
        type=positive_int,
        default=8,
        help="Maximum simultaneously active delivery orders",
    )
    parser.add_argument(
        "--uav-initial-energy",
        "--uav_initial_energy",
        dest="uav_initial_energy",
        type=float,
        default=100.0,
        help="Initial energy for UAVEnergyDelivery agents",
    )
    parser.add_argument(
        "--uav-energy-decay",
        "--uav_energy_decay",
        dest="uav_energy_decay",
        type=float,
        default=None,
        help="Energy consumed per UAVEnergyDelivery step",
    )
    parser.add_argument(
        "--uav-energy-depletion-fraction",
        "--uav_energy_depletion_fraction",
        dest="uav_energy_depletion_fraction",
        type=float,
        default=0.5,
        help="Fraction of episode length when full energy should be depleted",
    )
    parser.add_argument(
        "--uav-charging-capacity",
        "--uav_charging_capacity",
        dest="uav_charging_capacity",
        type=positive_int,
        default=2,
        help="Maximum UAVs charging at the station in one step",
    )
    parser.add_argument(
        "--uav-charging-radius",
        "--uav_charging_radius",
        dest="uav_charging_radius",
        type=float,
        default=0.18,
        help="Distance threshold for UAVEnergyDelivery charging station",
    )
    parser.add_argument(
        "--uav-charging-rate",
        "--uav_charging_rate",
        dest="uav_charging_rate",
        type=float,
        default=None,
        help="Energy restored per charging step",
    )
    parser.add_argument(
        "--save-xy",
        dest="save_xy",
        action="store_true",
        help="For UAV3D, save an XY-view GIF when using rgb_array render",
    )
    parser.add_argument(
        "--no-save-xy",
        dest="save_xy",
        action="store_false",
        help="Disable XY-view GIF output",
    )
    parser.add_argument(
        "--save-frames",
        dest="save_frames",
        action="store_true",
        help="Save each rendered 3D frame and top-view frame as PNG files",
    )
    parser.add_argument(
        "--no-save-frames",
        dest="save_frames",
        action="store_false",
        help="Disable per-frame PNG output",
    )
    parser.set_defaults(save_frames=True)
    parser.set_defaults(save_xy=True)
    parser.set_defaults(render=True)
    return parser.parse_args()


def make_base_args(cli_args):
    return SimpleNamespace(
        difficulty="7",
        game_version="latest",
        map=cli_args.map,
        sc2_path='D:/Program Files (x86)/StarCraft II',
        seed=123 if cli_args.seed is None else cli_args.seed,
        step_mul=8,
        replay_dir="",
        debug=False,
        alg=cli_args.alg,
        alg_list="",
        n_steps=1_000_000,
        n_episodes=1,
        evaluate_cycle=20,
        evaluate_epoch=1,
        last_action=False,
        reuse_network=True,
        gamma=0.99,
        optimizer="RMS",
        model_dir=cli_args.model_dir,
        result_dir="./result",
        load_model=True,
        evaluate=True,
        cuda=cli_args.cuda,
        gpu_id=cli_args.gpu_id,
        uav_n_agents=cli_args.uav_n_agents,
        episode_limit=cli_args.episode_limit,
        uav_total_orders=cli_args.uav_total_orders,
        uav_max_active_orders=cli_args.uav_max_active_orders,
        uav_initial_energy=cli_args.uav_initial_energy,
        uav_energy_decay=cli_args.uav_energy_decay,
        uav_energy_depletion_fraction=cli_args.uav_energy_depletion_fraction,
        uav_charging_capacity=cli_args.uav_charging_capacity,
        uav_charging_radius=cli_args.uav_charging_radius,
        uav_charging_rate=cli_args.uav_charging_rate,
        last_reward=False,
        distributed=True,
        guide_mix_network_type="vdn",
        comm=False,
        msg_size=3,
        aoi_threshold=0.25,
        comm_cost_penalty=0.1,
        comm_effect_bonus=0.1,
        comm_warning_threshold=0.1,
        comm_max_keep_dim=8,
        comm_lr=None,
        safety_lr=None,
        safety_beta=None,
        warning_penalty_weight=None,
        aoi_min_weight=0.2,
        aoi_stale_decay=0.5,
        comm_aoi_penalty=0.1,
        comm_warning_penalty=0.2,
        comm_fresh_bonus=0.05,
        now="test",
    )


def configure_algorithm_args(args):
    if args.alg.find("coma") > -1:
        args = get_coma_args(args)
    elif args.alg.find("central_v") > -1:
        args = get_centralv_args(args)
    elif args.alg.find("reinforce") > -1:
        args = get_reinforce_args(args)
    elif args.alg.find("mappo") > -1:
        args = get_mappo_args(args)
    elif args.alg.find("macpo") > -1:
        args = get_macpo_args(args)
    else:
        args = get_mixer_args(args)

    if args.alg.find("commnet") > -1:
        args = get_commnet_args(args)
    if args.alg.find("g2anet") > -1:
        args = get_g2anet_args(args)

    if args.alg.lower().find("comm") > -1 and args.alg.lower().find("rgmcomm") < 0:
        args.msg_shape = min(
            int(getattr(args, "comm_max_keep_dim", args.raw_obs_shape)),
            int(args.raw_obs_shape),
        )
        args.obs_shape += args.msg_shape * (args.n_agents - 1)
        args.state_shape = args.obs_shape * args.n_agents

    return args


def reset_env(env, seed=None):
    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        print(
            "Warning: environment reset does not accept seed; "
            "initialization may not be reproducible."
        )
        return env.reset()


def set_episode_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frame_dir(frame_root, alg, map_name, episode_idx, view):
    output_dir = os.path.join(
        frame_root,
        alg,
        map_name,
        f"episode_{episode_idx}",
        view,
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_frame_image(frame, frame_root, alg, map_name, episode_idx, view, frame_idx):
    output_dir = frame_dir(frame_root, alg, map_name, episode_idx, view)
    output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
    Image.fromarray(frame).save(output_path)
    return output_path


def render_frame(
    env,
    render_mode,
    frames,
    fps,
    xy_frames=None,
    save_xy=False,
    save_frames=False,
    frame_root=None,
    alg=None,
    map_name=None,
    episode_idx=0,
    frame_idx=0,
):
    if render_mode == "none":
        return

    frame = None
    xy_frame = None
    if hasattr(env, "env") and hasattr(env.env, "render"):
        frame = env.env.render(show=(render_mode == "human"))
        if save_xy or save_frames:
            xy_frame = env.env.render(show=False, view="xy")
    elif hasattr(env, "render"):
        frame = env.render()

    if render_mode == "human":
        if fps > 0:
            time.sleep(1.0 / fps)
        return

    if frame is not None:
        frames.append(Image.fromarray(frame))
        if save_frames:
            save_frame_image(
                frame,
                frame_root,
                alg,
                map_name,
                episode_idx,
                "3d",
                frame_idx,
            )
    if save_xy and xy_frame is not None and xy_frames is not None:
        xy_frames.append(Image.fromarray(xy_frame))
    if save_frames and xy_frame is not None:
        save_frame_image(
            xy_frame,
            frame_root,
            alg,
            map_name,
            episode_idx,
            "xy",
            frame_idx,
        )


def gif_path(result_root, alg, map_name, episode_idx, suffix=""):
    result_dir = os.path.join(result_root, alg, map_name)
    os.makedirs(result_dir, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    return os.path.join(result_dir, f"{map_name}_episode_{episode_idx}{suffix_part}.gif")


def print_model_info(args):
    model_path = os.path.join(args.model_dir, args.alg, args.map)
    print(f"Loaded model: {args.alg}/{args.map}")
    print(f"Model directory: {os.path.abspath(model_path)}")


def choose_actions(env, agents, args, last_action, step):
    raw_obs = np.asarray(env.get_obs(), dtype=np.float32)
    obs = raw_obs
    if getattr(agents, "use_comm_plugin", False) and args.alg.lower().find("rgmcomm") < 0:
        obs = agents.prepare_comm_obs(obs, 0.0)
        msg = None
    else:
        msg = _get_env_msg(env, args, args.n_agents)
    actions = []
    actions_onehot = []
    avail_actions = []
    for agent_id in range(args.n_agents):
        avail_action = env.get_avail_agent_actions(agent_id)
        action = agents.choose_action(
            obs[agent_id],
            last_action[agent_id],
            agent_id,
            avail_action,
            0.0,
            timestep_cur=step,
            timestep_max=args.n_steps,
            msg=None if msg is None else msg[agent_id],
        )
        action_onehot = np.zeros(args.n_actions, dtype=np.float32)
        action_onehot[action] = 1.0
        actions.append(int(action))
        actions_onehot.append(action_onehot)
        avail_actions.append(avail_action)
        last_action[agent_id] = action_onehot
    return raw_obs, actions, avail_actions, actions_onehot


def run_episode(env, agents, args, cli_args, episode_idx):
    if cli_args.seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        print(f"Episode {episode_idx} seed: {seed} (random)")
    else:
        seed = int(cli_args.seed + episode_idx)
        print(f"Episode {episode_idx} seed: {seed}")
    set_episode_seed(seed)
    reset_env(env, seed=seed)
    if hasattr(agents, "reset_episode_state"):
        agents.reset_episode_state()
    if hasattr(agents.policy, "init_hidden"):
        agents.policy.init_hidden(1)

    max_steps = cli_args.max_steps or args.episode_limit
    terminated = False
    step = 0
    episode_reward = 0.0
    last_action = np.zeros((args.n_agents, args.n_actions), dtype=np.float32)
    info = {
        "battle_won": False,
    }

    frames = []
    save_frame_images = bool(cli_args.save_frames and cli_args.render_mode == "rgb_array")
    xy_frames = (
        []
        if cli_args.render
        and cli_args.render_mode == "rgb_array"
        and cli_args.save_xy
        else None
    )
    render_mode = cli_args.render_mode if (cli_args.render or save_frame_images) else "none"
    render_frame(
        env,
        render_mode,
        frames,
        cli_args.fps,
        xy_frames=xy_frames,
        save_xy=bool(xy_frames is not None),
        save_frames=save_frame_images,
        frame_root=cli_args.frame_root,
        alg=args.alg,
        map_name=args.map,
        episode_idx=episode_idx,
        frame_idx=step,
    )

    while not terminated and step < max_steps:
        raw_obs, actions, avail_actions, _ = choose_actions(env, agents, args, last_action, step)
        if hasattr(agents, "revise_safe_actions"):
            revised_actions = agents.revise_safe_actions(
                observations=raw_obs,
                avail_actions=avail_actions,
                base_actions=actions,
            )
            if revised_actions is not None:
                actions = [int(action) for action in revised_actions]
        reward, terminated, info = env.step(actions)
        episode_reward += float(np.asarray(reward, dtype=np.float32).mean())
        step += 1
        render_frame(
            env,
            render_mode,
            frames,
            cli_args.fps,
            xy_frames=xy_frames,
            save_xy=bool(xy_frames is not None),
            save_frames=save_frame_images,
            frame_root=cli_args.frame_root,
            alg=args.alg,
            map_name=args.map,
            episode_idx=episode_idx,
            frame_idx=step,
        )

    win_tag = bool(terminated and info.get("battle_won", False))
    summary = _build_env_summary(env, info, step, win_tag, args.n_agents)
    summary["episode_reward"] = episode_reward

    if render_mode == "rgb_array" and frames:
        output_path = gif_path(cli_args.result_root, args.alg, args.map, episode_idx)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(1000 / cli_args.gif_fps)),
            loop=0,
        )
        print(f"Saved visualization to {output_path}")
        if xy_frames:
            xy_output_path = gif_path(
                cli_args.result_root, args.alg, args.map, episode_idx, suffix="xy"
            )
            xy_frames[0].save(
                xy_output_path,
                save_all=True,
                append_images=xy_frames[1:],
                duration=max(1, int(1000 / cli_args.gif_fps)),
                loop=0,
            )
            print(f"Saved XY visualization to {xy_output_path}")

    if save_frame_images:
        frame_base = os.path.join(
            cli_args.frame_root,
            args.alg,
            args.map,
            f"episode_{episode_idx}",
        )
        print(
            f"Saved per-frame images to {frame_base}/3d and {frame_base}/xy"
        )

    return summary


def main():
    cli_args = parse_args()
    if cli_args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(cli_args.gpu_id))
    args = make_base_args(cli_args)
    env = None

    try:
        env = build_env(args, [args.alg])
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.raw_obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.msg_shape = env_info.get("msg_shape", 0)
        args = configure_algorithm_args(args)

        agents = Agents(args, env)
        print_model_info(args)

        if cli_args.render:
            if cli_args.render_mode == "human":
                print(f"Render mode: human at about {cli_args.fps} FPS.")
            else:
                print(
                    f"Render mode: rgb_array, GIF output in "
                    f"{os.path.abspath(cli_args.result_root)}."
                )
                if cli_args.save_frames:
                    print(
                        "Per-frame 3D and XY images will be saved in "
                        f"{os.path.abspath(cli_args.frame_root)}."
                    )
        else:
            print("Render mode: none.")
            if cli_args.save_frames:
                print(
                    "Per-frame 3D and XY images will be saved in "
                    f"{os.path.abspath(cli_args.frame_root)}."
                )

        summaries = []
        for episode_idx in range(cli_args.episodes):
            summary = run_episode(env, agents, args, cli_args, episode_idx)
            summaries.append(summary)
            print(
                f"Episode {episode_idx}: reward={summary['episode_reward']:.3f}, "
                f"win={int(bool(summary['win_tag']))}, steps={int(summary['step'])}, "
                f"collisions={float(summary.get('collision_count', 0.0)):.1f}, "
                f"obstacle_collisions={float(summary.get('obstacle_collision_count', 0.0)):.1f}, "
                f"agent_collisions={float(summary.get('agent_collision_count', 0.0)):.1f}"
            )

        avg_reward = float(np.mean([summary["episode_reward"] for summary in summaries]))
        avg_win_rate = float(np.mean([summary["win_tag"] for summary in summaries]))
        avg_steps = float(np.mean([summary["step"] for summary in summaries]))
        avg_collision = float(
            np.mean([summary.get("collision_count", 0.0) for summary in summaries])
        )
        avg_obstacle_collision = float(
            np.mean(
                [summary.get("obstacle_collision_count", 0.0) for summary in summaries]
            )
        )
        avg_agent_collision = float(
            np.mean(
                [summary.get("agent_collision_count", 0.0) for summary in summaries]
            )
        )
        print(
            f"Average over {len(summaries)} episode(s): "
            f"reward={avg_reward:.3f}, win_rate={avg_win_rate:.3f}, steps={avg_steps:.1f}, "
            f"collisions={avg_collision:.1f}, obstacle_collisions={avg_obstacle_collision:.1f}, "
            f"agent_collisions={avg_agent_collision:.1f}"
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
