# Workspace consolidation — 2026-09-20

The workspace is now entirely under `/mnt/workspace/zjl-exp`. `repo` is a real directory, not a symlink. `/mnt/workspace/persistent-uav` has been removed. No active experiments or open file descriptors used the old tree at relocation.

The independent old tool session was preserved under `/mnt/workspace/zjl-exp/.preserved_sessions/persistent-uav-codex-session` without merging it into the active session. This private local directory contains authentication/session data; do not commit or upload it.

Thirteen virtualenv entry/activation scripts and `migration_20260920/env.sh` were updated to the new root. Original scripts and a hash audit are retained here. SSH keys remain unchanged under the workspace `.ssh` directory. No home-directory or global Git/SSH settings were changed.

Use `source /mnt/workspace/zjl-exp/migration_20260920/env.sh` for this environment. For any separately authorized future diagnostic, the new compatibility record is `migration_20260920/consolidation_20260920/calibration_compatibility_zjl_exp.json`; it verifies identical source/model/package hashes after explicit path relocation. No new experiment was started.

All 16 final snapshots deserialize and contain exactly 1080 completed results. Published evidence files have unchanged SHA256 hashes. Original manifests, snapshots, historical launch/resume commands and the previously uploaded report deliberately retain their original paths as evidence. They are not current launch instructions. Do not directly resume the completed validation or the superseded sequential/eight-worker runs. A future checkpoint continuation would require explicit contract relocation rather than blind text substitution.

The earlier path-repair handoff and uploaded results README describe the former layout; this document supersedes only their workspace-layout statements, not their experimental records.
