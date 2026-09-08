#!/usr/bin/env python3
"""Prepare by default; ONLY --start launches resumable HOCBF/SAC adaptation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.correction_checkpoint import load_correction_checkpoint
from experiments.directional_navigation.correction_sac import CorrectionSAC, warm_start
from experiments.directional_navigation.correction_supervision import (
    VERSION, CorrectionRecovery, TeacherConfig,
)
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.standard_baselines import atomic_json, save_checkpoint
from scripts.run_recovery_sac_ppo_scratch import Records, count_updates

STOP = False
DEFAULT_SOURCE = ROOT / "artifacts/recovery_sac_ppo_scratch_20260906_v1/sac/checkpoint_000524288/model.zip"


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--start", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--additional-steps", type=int, default=131072,
                        help="Adaptation transitions counted from zero; may extend on explicit resume")
    result.add_argument("--num-envs", type=int, default=8)
    result.add_argument("--rollout-steps", type=int, default=512)
    result.add_argument("--checkpoint-steps", type=int, default=32768)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--buffer-size", type=int, default=200000)
    result.add_argument("--learning-starts", type=int, default=5000)
    result.add_argument("--horizon", type=int, default=4000)
    result.add_argument("--obstacles", type=int, default=24)
    result.add_argument("--correction-weight", type=float, default=0.1,
                        help="0 is the HOCBF-only adaptation control")
    result.add_argument("--correction-batch-size", type=int, default=8)
    result.add_argument("--correction-interval", type=int, default=8)
    return result


def validate(args: argparse.Namespace) -> None:
    sizes = (args.additional_steps, args.num_envs, args.rollout_steps,
             args.checkpoint_steps, args.batch_size, args.buffer_size, args.horizon,
             args.correction_batch_size, args.correction_interval)
    if min(sizes) <= 0 or min(args.learning_starts, args.obstacles, args.seed) < 0:
        raise ValueError("invalid training sizes/seed")
    block = args.num_envs * args.rollout_steps
    if args.additional_steps % block or args.checkpoint_steps % block:
        raise ValueError("training/checkpoint budgets must align with complete blocks")
    if not np.isfinite(args.correction_weight) or args.correction_weight < 0:
        raise ValueError("nonnegative finite correction weight required")
    if args.correction_batch_size > args.batch_size:
        raise ValueError("correction batch must fit the SAC minibatch")
    if args.resume and not args.start:
        raise ValueError("resume requires explicit --start")
    if not args.source_model.is_file():
        raise ValueError("source checkpoint is missing")
    if stable_baselines3.__version__ != "2.8.0":
        raise ValueError("SAC update compatibility was audited against SB3 2.8.0")


def contract_for(args: argparse.Namespace) -> dict:
    files = [Path(__file__),
        ROOT / "experiments/directional_navigation/correction_supervision.py",
        ROOT / "experiments/directional_navigation/correction_sac.py",
        ROOT / "experiments/directional_navigation/correction_checkpoint.py",
        ROOT / "experiments/directional_navigation/features.py",
        ROOT / "experiments/directional_navigation/recovery.py",
        ROOT / "experiments/directional_navigation/environment.py",
        ROOT / "experiments/directional_navigation/standard_baselines.py",
        ROOT / "experiments/jacobian_energy_bridge/features.py",
        ROOT / "scripts/run_recovery_sac_ppo_scratch.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
        ROOT / "review_bundle/safety/collision/filter.py",
        ROOT / "review_bundle/safety/collision/hocbf.py",
        ROOT / "review_bundle/safety/collision/geometry3d.py",
        ROOT / "review_bundle/safety/collision/projection_geometry.py",
        ROOT / "review_bundle/safety/collision/_qp_native.c",
    ]
    native = ROOT / "review_bundle/safety/collision/_qp_native.so"
    if native.exists():
        files.append(native)
    arguments = {k: str(v.resolve()) if isinstance(v, Path) else v
                 for k, v in vars(args).items()
                 if k not in {"output_dir", "start", "resume", "additional_steps"}}
    return {"protocol": VERSION, "arguments": arguments,
        "source_model_sha256": hashlib.sha256(args.source_model.read_bytes()).hexdigest(),
        "observation_dim": 2056, "network": "unchanged_old_directional_SAC_2081_features",
        "teacher": asdict(TeacherConfig()),
        "supervision": "fresh_current_stochastic_action_projection_physical_squared_distance",
        "replay_semantics": "nominal_action_in_shielded_environment; new_replay_on_warm_start",
        "initialization": "copy_policy_critics_entropy; reset_optimizers_and_replay",
        "collision_contract": "locked_nonterminal_repair_unified_policy_step_count",
        "reward": "old_effective_reward; dormant_intervention_penalty_explicitly_zero",
        "hocbf": True, "cached_jacobians": False, "privileged_obstacles": False,
        "extra_critic": False, "energy_objective": False,
        "automatic_evaluation": False, "automatic_promotion": False,
        "sb3": stable_baselines3.__version__, "torch": torch.__version__,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files}}


def make_model(vec, args: argparse.Namespace) -> CorrectionSAC:
    source = SAC.load(args.source_model, device=args.device)
    extractor = source.actor.features_extractor
    if (not isinstance(extractor, DirectionalLidarExtractor)
            or not extractor.remaining_time or extractor.readout != "ordered"
            or extractor.features_dim != 2081):
        raise ValueError("source must be the old directional/time-aware SAC, not v2/v3/JSEB")
    model = CorrectionSAC("MlpPolicy", vec, learning_rate=3e-4, gamma=0.99,
        batch_size=args.batch_size, seed=args.seed, device=args.device,
        policy_kwargs=source.policy_kwargs, buffer_size=args.buffer_size,
        learning_starts=args.learning_starts, tau=0.005, train_freq=1,
        gradient_steps=-1, ent_coef="auto_0.01", target_entropy=-3.0,
        correction_weight=args.correction_weight,
        correction_batch_size=args.correction_batch_size,
        correction_interval=args.correction_interval)
    warm_start(model, source)
    del source
    return model


def train(args: argparse.Namespace, root: Path, contract: dict) -> None:
    global STOP
    STOP = False
    arm = root / "sac"
    arm.mkdir(exist_ok=True)
    if (arm / "latest.json").exists() != args.resume:
        raise ValueError("--resume must match existence of a committed checkpoint")
    if (root / "PAUSE").exists():
        raise ValueError("PAUSE still present; no training started")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    vec = SubprocVecEnv([partial(CorrectionRecovery,
        seed_start=993600001 + args.seed * 1000000 + i, seed_stride=args.num_envs,
        horizon=args.horizon, obstacles=args.obstacles)
        for i in range(args.num_envs)], start_method="forkserver")
    hooks = []
    try:
        if args.resume:
            model, records = load_correction_checkpoint(vec, arm, contract, args.device)
            if args.additional_steps < model.num_timesteps:
                raise ValueError("requested budget precedes restored checkpoint")
        else:
            model = make_model(vec, args)
            vec._reset_seeds()
            records = {"episodes": [], "updates": [], "training_seconds": 0.0,
                       "source_transitions": model.source_transitions}
        callback = Records(records)
        hooks = count_updates(model, records)
        block = args.num_envs * args.rollout_steps
        latest = model.num_timesteps if args.resume else -1
        while model.num_timesteps < args.additional_steps and not STOP and not (root / "PAUSE").exists():
            atomic_json(root / "status.json", {"status": "TRAINING", "pid": os.getpid(),
                "adaptation_transitions": model.num_timesteps, "target": args.additional_steps})
            started = time.perf_counter()
            callback.collect_seconds = 0.0
            model.learn(block, reset_num_timesteps=False, callback=callback, log_interval=None)
            elapsed = time.perf_counter() - started
            metrics = {k: float(v) for k, v in model.logger.name_to_value.items()
                       if k.startswith(("train/", "correction/"))
                       and isinstance(v, (int, float, np.number))}
            if (not all(np.isfinite(v) for v in metrics.values()) or not all(
                    torch.isfinite(p).all().item() for p in model.policy.parameters())):
                raise FloatingPointError("nonfinite training state")
            records["training_seconds"] += elapsed
            records["updates"].append({"transitions": model.num_timesteps,
                "block_seconds": elapsed, "collection_seconds": callback.collect_seconds,
                "metrics": metrics})
            atomic_json(arm / "training_records.json", records)
            pause = STOP or (root / "PAUSE").exists()
            if (latest < 0 or model.num_timesteps == 2 * block
                    or model.num_timesteps % args.checkpoint_steps == 0
                    or model.num_timesteps >= args.additional_steps or pause):
                save_checkpoint(model, vec, arm, contract, records)
                latest = model.num_timesteps
            atomic_json(root / "status.json", {"status": "TRAINING", "pid": os.getpid(),
                "adaptation_transitions": model.num_timesteps,
                "source_transitions": model.source_transitions,
                "checkpoint_transitions": latest, "training_seconds": records["training_seconds"],
                "correction_totals": model.correction_totals})
        if latest != model.num_timesteps:
            save_checkpoint(model, vec, arm, contract, records)
        atomic_json(root / "status.json", {
            "status": "TRAINING_COMPLETE_AWAITING_USER" if model.num_timesteps >= args.additional_steps else "PAUSED",
            "adaptation_transitions": model.num_timesteps, "automatic_evaluation": False,
            "checkpoint_transitions": model.num_timesteps,
            "correction_totals": model.correction_totals})
    finally:
        for hook in hooks:
            hook.remove()
        vec.close()


def main() -> None:
    args = parser().parse_args()
    validate(args)
    root, contract = args.output_dir.resolve(), contract_for(args)
    if root.exists():
        manifest = root / "manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text())["contract"] != contract:
            raise ValueError("output directory unrelated, or prepared source/config changed")
    else:
        if args.resume:
            raise ValueError("cannot resume missing run")
        root.mkdir(parents=True)
        atomic_json(root / "manifest.json", {"contract": contract,
            "prepared_unix": time.time(), "planned_additional_steps": args.additional_steps})
        atomic_json(root / "status.json", {"status": "READY_NOT_STARTED", "adaptation_transitions": 0})
    if not args.start:
        print(json.dumps({"status": "PREPARATION_ONLY", "training_started": False,
                          "output_dir": str(root), "planned_additional_steps": args.additional_steps}))
        return
    try:
        train(args, root, contract)
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
