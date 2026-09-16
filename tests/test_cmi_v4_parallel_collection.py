import json
from dataclasses import asdict
from pathlib import Path

from scripts.run_cmi_v4_parallel_collection import WorldTask, checkpoint, validate_ticket
from scripts.run_conditional_history_collection import atomic_json
from scripts.validate_identification_contract import file_hash


def _stage(root: Path, task: WorldTask) -> dict:
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    json_path = records / f"{task.index}.json"
    npz_path = records / f"{task.index}.npz"
    json_path.write_text("{}\n")
    npz_path.write_bytes(b"npz")
    record = {
        "world_identity": task.identity,
        "json": json_path.name,
        "npz": npz_path.name,
        "json_sha256": file_hash(json_path).removeprefix("sha256:"),
        "npz_sha256": file_hash(npz_path).removeprefix("sha256:"),
        "total_branch_policy_steps": task.index + 10,
    }
    ticket = {
        "schema_version": "cmi-v4-world-ticket-v1",
        "task": asdict(task),
        "record": record,
        "worker_pid": 123,
        "wall_seconds": 1.0,
        "role": "CMI_V4_FORMAL_RAW_WORLD",
    }
    atomic_json(root / "staged" / f"{task.index:04d}.json", ticket)
    return record


def test_checkpoint_commits_only_contiguous_tickets(tmp_path: Path) -> None:
    tasks = [WorldTask(i, f"sha256:{i:064x}", 100 + i) for i in range(3)]
    _stage(tmp_path, tasks[1])
    assert checkpoint(tmp_path, tasks, [], 2.0) == []
    first = _stage(tmp_path, tasks[0])
    committed = checkpoint(tmp_path, tasks, [], 3.0)
    assert [row["world_identity"] for row in committed] == [
        tasks[0].identity,
        tasks[1].identity,
    ]
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["generation"] == 2
    assert latest["total_branch_policy_steps"] == 21
    assert validate_ticket(tmp_path, tasks[0])["record"] == first


def test_ticket_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    task = WorldTask(0, "sha256:" + "a" * 64, 100)
    _stage(tmp_path, task)
    (tmp_path / "records" / "0.npz").write_bytes(b"changed")
    try:
        validate_ticket(tmp_path, task)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("corrupt ticket was accepted")
