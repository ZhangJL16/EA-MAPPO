"""Resumable, privileged *first-option* feasibility diagnostic.

Runs three cloned legal first actions from each original initial state.  It does
not run a scheduler for a full horizon, certify a safe policy, or edit the old
UAV project.  Each result is committed atomically after one initial sample.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import NamedTemporaryFile


PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT.parents[1] / "persistent_uav_throughput_v1"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from persistent_uav.baselines import Action  # noqa: E402
from persistent_uav.config import Config  # noqa: E402
from persistent_uav.environment import PersistentUAVThroughput  # noqa: E402
from persistent_uav.navigation import FrozenNavigator  # noqa: E402
from persistent_uav.provenance import provenance, verify_provenance  # noqa: E402
from persistent_uav.streams import workload  # noqa: E402

from collect_initial_choice_lb import validate_row  # noqa: E402


FROZEN = OLD / "evidence/v1_2/calibration/frozen_regimes.json"
SEEDS = tuple(range(1202609290, 1202609390))  # New; disjoint from old splits.


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def contract():
    frozen = json.loads(FROZEN.read_text())
    if frozen.get("pilot_jobs") != 1000 or not frozen.get("pilot_complete"):
        raise RuntimeError("frozen calibration is incomplete")
    groups = {}
    for row in frozen["regimes"]:
        capacity = row["config"]["capacity"]
        groups.setdefault(capacity, []).append(row)
    if len(groups) != 3 or any(len(rows) != 9 for rows in groups.values()):
        raise RuntimeError("expected three capacity groups of nine frozen regimes")
    capacity_groups = []
    first_option_fields = ("capacity", "cutoff", "queue_capacity", "initial_tasks",
                           "option_step_limit", "delta")
    for capacity, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: x["id"])
        representative = rows[0]
        representative_config = Config(**representative["config"])
        common = {key: getattr(representative_config, key) for key in first_option_fields}
        for row in rows:
            actual_config = Config(**row["config"])
            if {key: getattr(actual_config, key) for key in first_option_fields} != common:
                raise RuntimeError(f"first-option config differs within capacity {capacity}")
        # Exhaustively audit the actual initial three-task generator over the
        # predeclared seed set, rather than treating 9 regime labels as samples.
        initial_rows = []
        for seed in SEEDS:
            positions = [list(task.position) for task in workload(
                seed, Config(**representative["config"]), frozen["layout"])[:3]]
            for row in rows[1:]:
                other = [list(task.position) for task in workload(
                    seed, Config(**row["config"]), frozen["layout"])[:3]]
                if other != positions:
                    raise RuntimeError(f"initial task positions differ: capacity={capacity}, seed={seed}")
            initial_rows.append(dict(seed=seed, positions=positions))
        positions_sha = sha256(json.dumps(initial_rows, sort_keys=True).encode()).hexdigest()
        capacity_groups.append(dict(
            capacity=capacity, regime_ids=[x["id"] for x in rows],
            representative_regime=representative["id"], config=representative["config"],
            common_first_option_fields=common,
            initial_positions_sha256_all_declared_seeds=positions_sha))
    record = dict(
        kind="initial_choice_unavoidable_depletion_lb_v1",
        scope="original initial distribution and first nonpreemptive service only",
        target="Z=1 iff all three legal first serve actions physically deplete before reach, timeout, or cutoff",
        interpretation="privileged cloned-branch diagnostic; no full-T policy risk certificate",
        frozen_calibration=str(FROZEN), frozen_sha256=sha(FROZEN),
        runner_sha256=sha(__file__), capacity_groups=capacity_groups,
        seeds=list(SEEDS), n_per_capacity=len(SEEDS),
        provenance=provenance(),
        rule="if any branch does not deplete, Z=0 and remaining branches are skipped",
        risk_alpha_total=0.05,
        separate_group_claims="exact binomial one-sided lower test, Bonferroni alpha=0.05/3",
        safety_boundary="simulator distribution only; no physical aircraft transfer",
    )
    return record, frozen


def ensure_manifest(output, record, *, init_only):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != record:
            raise RuntimeError("resume contract mismatch; use a new output directory")
    elif not init_only:
        raise RuntimeError("manifest absent; run --init-only before workers")
    else:
        if any(output.iterdir()):
            raise RuntimeError("new output directory is not empty")
        write_json_atomic(path, record)


def first_branch(config, layout, tasks, task_id):
    nav = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit,
                          layout=layout)
    try:
        env = PersistentUAVThroughput(config, nav, tasks)
        observation = env.observe()
        assert env.decision_required and not env.can_recharge
        assert len(observation["queue"]) == 3
        assert abs(observation["battery"] - config.capacity) < 1e-6
        assert {t["id"] for t in observation["queue"]} == {0, 1, 2}
        first = True
        while True:
            env.step(Action("serve", task_id, "initial_choice_lb") if first else None)
            first = False
            if env.done or env.mode != "SERVING":
                break
        assert env.completed in (0, 1)
        if env.failure == "energy_depletion":
            outcome = "energy_depletion"
        elif env.failure == "navigation_failure":
            outcome = "navigation_failure"
        elif env.completed == 1:
            outcome = "completed_first_service"
        elif env.done and env.mode == "CUTOFF":
            outcome = "cutoff_before_first_completion"
        else:
            raise RuntimeError(f"unrecognized first-option endpoint: {env.mode}")
        return dict(task_id=task_id, outcome=outcome,
                    endpoint_time=env.time, endpoint_battery=nav.energy,
                    policy_steps=nav.policy_steps, collision_count=nav.contacts,
                    failure=env.failure, completed=env.completed,
                    stream_sha256=env.workload_hash)
    finally:
        nav.close()


def one_job(group, seed, frozen, record_hash):
    config = Config(**group["config"])
    tasks = workload(seed, config, frozen["layout"])
    if tuple(t.id for t in tasks[:3]) != (0, 1, 2):
        raise RuntimeError("initial task ID contract changed")
    branches = []
    for task_id in (0, 1, 2):
        branch = first_branch(config, frozen["layout"], tasks, task_id)
        branches.append(branch)
        if branch["outcome"] != "energy_depletion":
            break
    return dict(contract_sha256=record_hash, capacity=group["capacity"],
                representative_regime=group["representative_regime"],
                regime_ids=group["regime_ids"], seed=seed,
                initial_task_positions=[list(t.position) for t in tasks[:3]],
                branches=branches, skipped_task_ids=list(range(len(branches), 3)),
                all_legal_first_actions_deplete=(len(branches) == 3 and all(
                    x["outcome"] == "energy_depletion" for x in branches)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/initial_choice_lb_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--stop-after-jobs", type=int, default=0)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("invalid worker index/count")
    if args.stop_after_jobs < 0:
        parser.error("stop-after-jobs must be nonnegative")
    record, frozen = contract()
    verify_provenance(record["provenance"])
    ensure_manifest(args.output, record, init_only=args.init_only)
    if args.init_only:
        return
    record_hash = sha(args.output / "manifest.json")
    jobs = [(group, seed) for group in record["capacity_groups"] for seed in SEEDS]
    selected = [(i, *job) for i, job in enumerate(jobs) if i % args.workers == args.worker_index]
    stopping = [False]

    def ask_stop(signum, frame):
        stopping[0] = True

    signal.signal(signal.SIGTERM, ask_stop)
    signal.signal(signal.SIGINT, ask_stop)
    completed_this_call = 0
    for index, group, seed in selected:
        row_path = args.output / "rows" / f"job_{index:04d}.json"
        if row_path.exists():
            previous = json.loads(row_path.read_text())
            validate_row(row_path, previous, record, frozen, record_hash, index, group, seed)
            continue
        row = one_job(group, seed, frozen, record_hash)
        write_json_atomic(row_path, row)
        completed_this_call += 1
        status = dict(worker_index=args.worker_index, workers=args.workers,
                      last_job_index=index, last_seed=seed,
                      completed_this_call=completed_this_call,
                      saved_row=str(row_path), stopping=stopping[0])
        write_json_atomic(args.output / f"status_worker_{args.worker_index:02d}.json", status)
        print(json.dumps(dict(event="checkpoint", **status)), flush=True)
        if stopping[0] or (args.stop_after_jobs and completed_this_call >= args.stop_after_jobs):
            return
    write_json_atomic(args.output / f"status_worker_{args.worker_index:02d}.json",
                      dict(worker_index=args.worker_index, workers=args.workers,
                           complete=True, assigned_jobs=len(selected)))


if __name__ == "__main__":
    main()
