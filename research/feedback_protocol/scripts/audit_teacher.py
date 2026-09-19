"""Replay every completed TRAIN label and optionally export portable evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fpl.teacher_data import audit_label, coverage, build_tasks
from fpl.evaluation import digest, source_hashes, save_new
from fpl.problem import load_problem
from fpl.generator import problem_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="new audit directory")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.output)
    if out.exists():
        raise FileExistsError("do not overwrite a previous audit")
    seal = json.loads((data / "seal.json").read_text())
    if seal["sources"] != source_hashes():
        raise ValueError("audit runtime differs from sealed runtime")
    jobs, rejected = build_tasks(seal["plan"], seal["exclusions"])
    if seal["jobs"] != [dict(instance_hash=p.instance_hash, metadata=m) for p,m in jobs] or rejected != seal["rejections"]:
        raise ValueError("admission replay mismatch")
    expected = {p.instance_hash:(p,m) for p,m in jobs}
    shards, files = [], [data / "seal.json", data / "coverage.json"]
    for path in sorted(data.glob("*.shard.json")):
        envelope = json.loads(path.read_text())
        shard = envelope["payload"]
        if digest(shard) != envelope["hash"] or shard["instance_hash"] not in expected:
            raise ValueError("shard checksum/identity mismatch")
        p, meta = expected[shard["instance_hash"]]
        if shard["metadata"] != meta or shard["problem"] != json.loads(problem_json(p)):
            raise ValueError("public problem mismatch")
        if path.name != f"{p.instance_hash}.shard.json":
            raise ValueError("filename identity mismatch")
        if len({r["id"] for r in shard["records"]}) != len(shard["records"]):
            raise ValueError("duplicate state label")
        for record in shard["records"]:
            audit_label(p, record)
        shards.append(shard)
        files.append(path)
    summary = coverage(shards, len(jobs))
    summary["rejected_candidates"] = len(rejected)
    summary["exhausted_strata"] = sum(bool(x.get("unresolved")) for x in rejected)
    if json.loads((data / "coverage.json").read_text()) != summary:
        raise ValueError("coverage replay mismatch")
    out.mkdir(parents=True)
    archive = out / "teacher_startup_evidence.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, "data/"+path.name)
        for relative in seal["sources"]:
            z.write(ROOT / "src/fpl" / relative, "runtime/fpl/"+relative)
        z.write(Path(__file__), "scripts/audit_teacher.py")
    save_new(out / "audit.json", dict(status="passed", scope="all completed shards; arithmetic, reachability, completeness and seal replay",
                                      limitation="not an independent proof of ExactBayes or a sufficient-training-coverage guarantee",
                                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), summary=summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
