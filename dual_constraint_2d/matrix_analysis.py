"""Paired, map-level analysis of the frozen static-full-map matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train_ppo import TEST_MAP_IDS, TRAIN_SEEDS, VALIDATION_MAP_IDS


METHODS = ("route_full", "route_partial", "mpc_h1", "mpc_h2")


def _read_split(root: Path, split: str, map_ids: tuple[int, ...]) -> dict:
    rows = {}
    for method in (*METHODS, *(f"ppo_seed_{seed}" for seed in TRAIN_SEEDS)):
        rows[method] = []
        for map_id in map_ids:
            path = root / split / method / f"map_{map_id:03d}" / "summary.json"
            if not path.exists():
                raise RuntimeError(f"incomplete matrix: {path}")
            row = json.loads(path.read_text())
            if not row["done"] or row["map_id"] != map_id:
                raise RuntimeError(f"invalid completed summary: {path}")
            rows[method].append(row)
    return rows


def _metrics(rows: list[dict]) -> dict:
    numeric = ("completed_targets", "collision_count", "safety_cost",
               "charge_events", "return_takeovers", "energy_filter_events",
               "collision_intervention_steps", "policy_decisions", "plant_decisions",
               "simulated_seconds")
    result = {name: float(np.mean([row[name] for row in rows])) for name in numeric}
    result["depletion_events"] = sum(bool(row["actual_depletion"]) for row in rows)
    result["return_failure_events"] = sum(
        bool(row["failure_reason"]) and row["failure_reason"] != "depletion"
        for row in rows
    )
    if "mpc_candidates" in rows[0]:
        result["mpc_candidates_total"] = sum(row["mpc_candidates"] for row in rows)
        result["mpc_wall_seconds_total"] = sum(row["mpc_wall_seconds"] for row in rows)
    return result


def _paired_interval(differences: np.ndarray) -> dict:
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError("paired interval requires at least two maps")
    rng = np.random.default_rng(20260926)
    indexes = rng.integers(0, len(differences), size=(10000, len(differences)))
    means = differences[indexes].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(np.mean(differences)),
        "bootstrap_95_percentile": [float(low), float(high)],
        "map_differences": differences.tolist(),
        "positive_maps": int(np.sum(differences > 0)),
        "tied_maps": int(np.sum(differences == 0)),
        "negative_maps": int(np.sum(differences < 0)),
    }


def analyze(root: Path) -> dict:
    validation = _read_split(root, "validation", VALIDATION_MAP_IDS)
    test = _read_split(root, "test", TEST_MAP_IDS)
    # Choose once using validation data.  A tie favors the deeper planner.
    validation_h1 = np.mean([r["completed_targets"] for r in validation["mpc_h1"]])
    validation_h2 = np.mean([r["completed_targets"] for r in validation["mpc_h2"]])
    comparator = "mpc_h1" if validation_h1 > validation_h2 else "mpc_h2"
    ppo_per_map = np.asarray([
        np.mean([test[f"ppo_seed_{seed}"][i]["completed_targets"]
                 for seed in TRAIN_SEEDS])
        for i in range(len(TEST_MAP_IDS))
    ])
    mpc_per_map = np.asarray([row["completed_targets"] for row in test[comparator]])
    difference = _paired_interval(ppo_per_map - mpc_per_map)
    method_metrics = {name: _metrics(rows) for name, rows in test.items()}
    ppo_safety = {
        key: sum(method_metrics[f"ppo_seed_{seed}"][key] for seed in TRAIN_SEEDS)
        for key in ("depletion_events", "return_failure_events")
    }
    ppo_safety["collision_count"] = sum(
        row["collision_count"] for seed in TRAIN_SEEDS
        for row in test[f"ppo_seed_{seed}"]
    )
    mpc_safety = {
        "depletion_events": method_metrics[comparator]["depletion_events"],
        "return_failure_events": method_metrics[comparator]["return_failure_events"],
        "collision_count": sum(row["collision_count"] for row in test[comparator]),
    }
    # Three PPO seeds create 48 episodes versus the MPC's 16.  Compare
    # empirical rates, not raw totals, while retaining raw totals above.
    safety_rate_not_worse = all(
        ppo_safety[key] / (len(TEST_MAP_IDS) * len(TRAIN_SEEDS))
        <= mpc_safety[key] / len(TEST_MAP_IDS)
        for key in mpc_safety
    )
    c1_supported = (
        difference["bootstrap_95_percentile"][0] > 0
        and safety_rate_not_worse
    )
    result = {
        "protocol": "dual_constraint_2d_static_fullmap_matrix_v0",
        "validation_mpc_mean_completed": {
            "mpc_h1": float(validation_h1), "mpc_h2": float(validation_h2),
        },
        "validation_selected_comparator": comparator,
        "test_maps": list(TEST_MAP_IDS),
        "test_seeds": list(TRAIN_SEEDS),
        "test_method_metrics": method_metrics,
        "ppo_minus_selected_mpc_completed": difference,
        "ppo_safety_event_totals_over_48_episodes": ppo_safety,
        "selected_mpc_safety_event_totals_over_16_episodes": mpc_safety,
        "ppo_safety_event_rates_not_worse": bool(safety_rate_not_worse),
        "c1_empirically_supported_under_prespecified_rule": bool(c1_supported),
        "interpretation_limit": (
            "Synthetic deterministic full-map evidence only.  The MPC is a bounded "
            "depth-1/2 controller, not a global optimum.  Zero safety events do not "
            "certify population risk.  PPO and MPC have different numbers of test episodes."
        ),
    }
    (root / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 二维静态全图矩阵结果",
        "",
        f"验证集预选 MPC：`{comparator}`；H1/H2 平均完成数分别为 "
        f"{validation_h1:.3f}/{validation_h2:.3f}。",
        f"测试集 PPO－预选 MPC 配对完成数均值：{difference['mean']:.3f}；"
        f"地图级 bootstrap 95% 分位区间："
        f"[{difference['bootstrap_95_percentile'][0]:.3f}, "
        f"{difference['bootstrap_95_percentile'][1]:.3f}]。",
        f"预设 C1 支持判定：{'是' if c1_supported else '否'}。",
        "",
        "| 方法 | 平均完成数 | 碰撞均值 | 电量耗尽次数 | 返站失败次数 | 平均补能次数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in method_metrics.items():
        lines.append(
            f"| {name} | {values['completed_targets']:.3f} | "
            f"{values['collision_count']:.3f} | {values['depletion_events']} | "
            f"{values['return_failure_events']} | {values['charge_events']:.3f} |"
        )
    lines.extend(["", result["interpretation_limit"], "",
                  "原始 manifest、检查点、事件日志和各地图 summary 位于同目录。", ""])
    (root / "RESULT.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
