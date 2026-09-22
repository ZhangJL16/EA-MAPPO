# Complete raw mechanism sweep results

All 2164 raw result files are included here: 2094 new gzip JSON traces and 70 imported C1 JSON traces. Every archive and member has a SHA256 in manifest.json; all members were extracted in memory and verified before publication. Original bytes, events and nested oracle evidence are preserved.

Archives contain repository-relative paths. From a repository checkout root, restore with:

```bash
for archive in persistent_uav_throughput_v1/evidence/mechanism_intervention_sweep_v1/raw_archives/raw_results_*.tar.gz; do
    tar -xzf "$archive"
done
```

Extract into a fresh checkout to avoid overwriting local result files. The manifest maps each job ID to its archive/member; imported C1 paths are under atlas_infinite_queue_v1. Historical absolute paths in continuations.json can be resolved using these repository-relative mappings. Simulator checkpoints and external runtime dependencies are not part of this raw-result bundle.
