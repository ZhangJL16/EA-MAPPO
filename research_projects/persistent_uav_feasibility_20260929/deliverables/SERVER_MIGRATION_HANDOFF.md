# ARM CPU 服务器交接：当前持续任务调度路线

日期：2026-09-29。服务器由用户和服务器端 Codex 管理；本机没有直接 SSH 地址，当前执行环境的 `.git` 只读，故本交接**不是已推送到 GitHub 或已在服务器启动**的证明。

## 当前研究与计算

原始题目是冻结 SAC 导航的单 UAV 持续随机任务流，27 个冻结 regime，完整 `T≈10905.05s` 内最大化完成任务数并评估实际耗尽风险。新研究在 `research_projects/persistent_uav_feasibility_20260929/`；旧物理与历史证据在 `persistent_uav_throughput_v1/`，只读。首任务共同耗尽下界 300/300 已完成，三容量档均 `Z=0/100`，这**不**是整窗安全证书。当前 810 条完整窗口固定策略筛查在本机运行，服务器不要同时用相同 manifest/seed 重复启动；完成后单独转移并校验原始结果。

当前 runner 的低层每 0.05s 使用 Python/NumPy 物理、LiDAR/CBF 和 SAC 单步前向；`persistent_uav_throughput_v1/persistent_uav/navigation.py` 用 `map_location='cpu'` 加载 actor 并在 CPU 推理。现有测试因而主要受 CPU 单步和多进程并行限制。用户提供的 ARM 服务器为 aarch64、40 CPU 核、约 229 GiB RAM，可先用 8–16 个进程实测速率和 RSS，再决定是否增加；Ascend 910 不会自动加速此实现。服务器曾安装 Python 3.12/uv、torch CPU、Gymnasium、SB3 等；需在运行当日核验版本，不能把旧查询当作现在状态。

## 迁移包与路径

`migration/persistent_uav_current_route_20260929.tar.gz` 只含当前项目、旧物理/完整历史结果、必需导航源码和模型、清理记录、必要文献索引；排除 `.venv`、Git、其他研究路线及**正在变化的** `runs/full_window_pilot_v1/`。文件级 SHA 清单为 `migration/bundle_manifest.json`，压缩包整体 SHA 在本机打包输出中记录。解包后应保持 `research_projects/` 与 `persistent_uav_throughput_v1/` 同级，`review_bundle` 相对符号链接指向 `runtime_support/review_bundle`。

`navigation.py` 的 `LEGACY` 默认是 `/home/zjl/mappo`，服务器需设置 `PERSISTENT_UAV_LEGACY_ROOT` 为解包后的仓库根目录。`provenance()` 还记录绝对文件路径和 NumPy/Torch/Gymnasium/SB3/SciPy 精确版本。**不能把本机进行中的 x86 manifest 直接拿到 ARM 续跑。** 服务器端若改变路径、依赖或源码，应在新的输出目录生成自己的 manifest，先核对 frozen actor SHA、27 regime SHA、真实导航 smoke 和跨平台行为；如需合并同一固定 seed 的 x86/ARM 结果，先做跨平台回放等价与兼容记录，否则分开报告。服务器端不应把硬件/依赖迁移称为原方法收益。

## 服务器端最小核对

在解包后的仓库根目录，一行设置变量并运行当前路径 smoke：`PERSISTENT_UAV_LEGACY_ROOT="$PWD" MPLCONFIGDIR=/tmp/mpl-uav /path/to/python research_projects/persistent_uav_feasibility_20260929/checks/smoke_full_window_pilot.py`。本机同一保留依赖集的隔离副本已经通过 50 个 policy steps 的真实导航 smoke；ARM 端仍须自行核验。完整窗口运行在本机完成前不要重复同一 810-job split。后续新 split 的抽样、风险认证和强同信息规划对照遵循 `board/posts/A02.md` 与 `deliverables/PRIOR_ROUND_TEST_MATRIX.md`，当前没有神经训练或 5% 风险认证。

## 清理状态

2026-09-29 用户选择直接删除当前仓库中非本路线的历史研究、旧归档与无关论文库。可审计记录在根目录 `docs/cleanup_20260929/`；哈希记录不是备份。保留的当前原文方法参考在 `papers/method_references/`，尚未由本轮逐篇全文复核。根目录 `README.md`、`PROJECT_CLOSEOUT.md`、`task_plan.md`、`notes.md` 已改为当前入口和历史边界，不再指向已删除归档。
