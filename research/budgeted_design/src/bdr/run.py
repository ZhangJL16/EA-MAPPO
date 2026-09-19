"""Bounded, resumable developer training/evaluation, never touches FPL data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .core import coupled_policy, random_inputs, objective, rollout, information_samples


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def binding(config):
    return {"config": config, "runtime": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "torch": torch.__version__, "python": platform.python_version()}


def load_checkpoint_policy(path, config, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["step"] != config["steps"]:
        raise ValueError("Only predetermined final checkpoint, not best-on-eval")
    policy = coupled_policy(saved["kind"], saved["seed"], config["hidden"], config["encoding"]).to(device)
    policy.load_state_dict(saved["state_dict"], strict=True)
    return policy.eval(), saved


def train(config, output, device, max_updates=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = binding(config)
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["binding"] != identity:
            raise ValueError("Resume source/config/runtime mismatch; use a new run directory")
    else:
        atomic_json(manifest, {"binding": identity, "device": str(device),
            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "DEV prototype; no final confirmation; no teacher"})
    started, updates = time.monotonic(), 0
    for seed in config["seeds"]:
        # Rotate execution order to reduce systematic warm-up/thermal effects.
        kinds = config["methods"]
        shift = config["seeds"].index(seed) % len(kinds)
        for kind in kinds[shift:] + kinds[:shift]:
            path = output / f"{kind}_{seed}.pt"
            policy = coupled_policy(kind, seed, config["hidden"], config["encoding"]).to(device)
            optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"])
            step, elapsed, log, measurements, likelihood_calls = 0, 0., [], 0, 0
            if path.exists():
                saved = torch.load(path, map_location=device, weights_only=False)
                policy.load_state_dict(saved["state_dict"])
                optimizer.load_state_dict(saved["optimizer"])
                step, elapsed, log = saved["step"], saved["elapsed_seconds"], saved["log"]
                measurements, likelihood_calls = saved["measurements"], saved["likelihood_calls"]
            while step < config["steps"]:
                if max_updates is not None and updates >= max_updates:
                    return
                if time.monotonic() - started > config["hard_timeout_seconds"]:
                    raise TimeoutError("Run wall-time watchdog; completed checkpoints retained")
                horizon = config["native_horizon"] if kind == "dad_native" else config["train_horizons"][step % len(config["train_horizons"]) ]
                sync(device)
                begin = time.perf_counter()
                inputs = random_inputs(seed * 100000 + step, config["batch"], horizon, config["contrasts"], device)
                optimizer.zero_grad(set_to_none=True)
                loss = -objective(policy, inputs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss {kind}/{seed}/{step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                sync(device)
                elapsed += time.perf_counter() - begin
                step += 1
                updates += 1
                measurements += config["batch"] * horizon
                likelihood_calls += config["batch"] * horizon * (config["contrasts"] + 1)
                log.append({"step": step, "H": horizon, "spce": -loss.item(), "grad_norm_before_clip": float(norm)})
                if step % config["checkpoint_every"] == 0 or step == config["steps"] or (max_updates is not None and updates >= max_updates):
                    saved = {"kind": kind, "seed": seed, "step": step, "state_dict": policy.state_dict(),
                        "optimizer": optimizer.state_dict(), "elapsed_seconds": elapsed, "log": log,
                        "measurements": measurements, "likelihood_calls": likelihood_calls,
                        "parameters": sum(p.numel() for p in policy.parameters()), "binding": identity}
                    temp = Path(str(path) + ".tmp")
                    torch.save(saved, temp)
                    os.replace(temp, path)
                    print(json.dumps({"kind": kind, "seed": seed, "step": step,
                        "mean_spce_last50": float(np.mean([x["spce"] for x in log[-50:]])), "seconds": elapsed}), flush=True)


@torch.no_grad()
def latency(policy, device):
    history = random_inputs(555, 1, 8, 1, device)[1].new_zeros(1, 4, 3)
    # Fixed public history benchmark; no simulator or evaluator in timed region.
    for _ in range(20):
        policy(history, 4)
    values = []
    for _ in range(100):
        sync(device)
        start = time.perf_counter()
        policy(history, 4)
        sync(device)
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.quantile(values, .9)),
        "scope": "batch=1, history length=4, remaining=4; synchronized if GPU"}


@torch.no_grad()
def evaluate(config, output, device):
    output = Path(output)
    manifest_name = "evaluation_manifest.json" if (output / "evaluation_manifest.json").exists() else "manifest.json"
    if json.loads((output / manifest_name).read_text())["binding"] != binding(config):
        raise ValueError("Source/config mismatch")
    all_rows = []
    start = time.monotonic()
    for seed in config["seeds"]:
        for kind in config["methods"]:
            policy, saved = load_checkpoint_policy(output / f"{kind}_{seed}.pt", config, device)
            variants = [("normal", None)]
            if kind == "budget_attention":
                variants += [("query_constant_4", 4), ("query_zero", 0)]
            for intervention, override in variants:
                for horizon in config["eval_horizons"]:
                    key = f"eval_{kind}_{seed}_{intervention}_H{horizon}.json"
                    path = output / key
                    if path.exists():
                        all_rows.append(json.loads(path.read_text()))
                        continue
                    if time.monotonic() - start > config["hard_timeout_seconds"]:
                        raise TimeoutError("Evaluation watchdog")
                    lowers, uppers = [], []
                    begin = time.perf_counter()
                    for block, offset in enumerate(range(0, config["eval_rollouts"], config["eval_batch"])):
                        batch = min(config["eval_batch"], config["eval_rollouts"] - offset)
                        # Same independent rollout/noise/contrastive blocks for ALL methods/seeds.
                        inputs = random_inputs(config["eval_seed"] + horizon * 1000 + block, batch, horizon, config["eval_contrasts"], device)
                        theta, noise, inner = inputs
                        history = rollout(policy, theta, noise, override)
                        lower, upper = information_samples(history, theta, inner)
                        lowers.extend(lower.cpu().tolist())
                        uppers.extend(upper.cpu().tolist())
                    row = {"kind": kind, "seed": seed, "intervention": intervention, "H": horizon,
                        "lower_samples": lowers, "upper_samples": uppers,
                        "lower_mean": float(np.mean(lowers)), "upper_mean": float(np.mean(uppers)),
                        "lower_mc_se": float(np.std(lowers, ddof=1) / np.sqrt(len(lowers))),
                        "upper_mc_se": float(np.std(uppers, ddof=1) / np.sqrt(len(uppers))),
                        "evaluation_seconds": time.perf_counter() - begin,
                        "checkpoint_sha256": digest(output / f"{kind}_{seed}.pt")}
                    atomic_json(path, row)
                    all_rows.append(row)
                    print(json.dumps({k:row[k] for k in ["kind", "seed", "intervention", "H", "lower_mean", "upper_mean"]}), flush=True)
            cost_path = output / f"cost_{kind}_{seed}.json"
            if not cost_path.exists():
                gpu_latency = latency(policy, device)
                policy.cpu()
                cpu_latency = latency(policy, "cpu")
                atomic_json(cost_path, {"kind":kind, "seed":seed, "parameters":saved["parameters"],
                    "training_seconds":saved["elapsed_seconds"], "measurements":saved["measurements"],
                    "likelihood_calls":saved["likelihood_calls"], "cpu_latency":cpu_latency,
                    "training_device_latency":gpu_latency, "training_device":str(device),
                    "online_likelihood_calls":0, "online_simulator_calls":0,
                    "note":"zero model calls is not free computation; see latency/parameters"})
    atomic_json(output / "evaluation.json", all_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-updates", type=int)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if args.mode == "train":
        train(config, args.output, args.device, args.max_updates)
    else:
        evaluate(config, args.output, args.device)


if __name__ == "__main__":
    main()
