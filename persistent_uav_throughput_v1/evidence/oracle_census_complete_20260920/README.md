# 完整 Oracle Recoverability Census

固定的 205 个历史状态、561 条分支已于 **2026-09-20 23:10:30（Asia/Shanghai）** 全部完成。
16/16 workers complete，无 error 文件；本次按用户指令执行既定 manual collect，未运行新 simulation。
运行源码为 `2da2125`，启动证据为 `6431d17`。collect 校验 frozen plan、实现和 runtime 哈希、
205 个完整历史 root 的 observation/event prefix、所有分支文件与预定 branch specs。
完整性 PASS：205 unique states、561 unique branches，无缺失状态、无重复状态。

## 状态结果

这里的“安全”仅指冻结 controller 下该次 bounded task→immediate return / direct return 成功。

| 组 | 状态 | 有安全任务 | 无安全任务但立即返航成功 | 无任何测试成功 continuation |
|---|---:|---:|---:|---:|
| A：full-station fallback | 67 | 12 | 0（返航动作不合法） | 55 |
| B：empty-queue pre-idle | 48 | 不适用 | 47（只测试返航） | 1 |
| C：task-error / task depletion | 90 | 59 | 28 | 3 |
| 总计 | 205 | 71 | 75 | 59 |

互斥分区与逐状态 ID 见 `state_analysis.json`；原始 collect 结果见 `results.json`、`summary.json`。
A/C 中有安全替代任务的状态为 69 个（A=10，C=59）。A 另有 2 个原选任务加立即返航成功，
不能把这 2 个当作“选错任务”的证据。C 全部 90 个原选 task+immediate-return 均失败。
因此在 A/C **155 个原选 continuation 失败的状态**中，69/155=44.52% 存在安全替代任务。
这是所选失败前驱状态的条件比例，不是全体决策的发生率。

C 的交叉分类进一步显示：37 个状态有安全任务且返航成功；20 个有安全任务但直接返航不合法；
2 个有安全任务而直接返航失败；28 个只能通过直接返航成功；3 个没有测试成功动作且返航不合法。
这些结果支持“当前任务选择会影响 bounded continuation 的结果”，尚不证明长期规划优于局部筛选。

## 等待反事实

B 中 40/40 waiting-margin-loss 状态立即返航成功，8 个 waiting-depletion 状态中 7 个成功、1 个耗尽。
合计 47/48=97.92%。这支持在这些历史状态避免继续 idle；没有测量修改后的完整 policy 的吞吐量或风险。
A 的 2 个原任务加立即返航成功也说明，历史最终失败不能自动归因于该任务本身不安全。

## 分支结果与解释边界

| 组 | 分支 | 成功 | 电量耗尽 | Navigation timeout | Horizon censored |
|---|---:|---:|---:|---:|---:|
| A | 182 | 12 | 170 | 0 | 0 |
| B | 48 | 47 | 1 | 0 | 0 |
| C | 331 | 151 | 180 | 0 | 0 |
| 总计 | 561 | 210 | 351 | 0 | 0 |

同一 root 的分支相关，不能视为 561 个独立 episode 风险样本。
A 的 55/67 状态没有测试成功任务，说明当前合法 bounded continuation 存在严重可行性问题，
**不证明整个 regime 或物理任务不可行**；未枚举多任务路线或其他 controller。
B 的 47 个与 C 的 28 个状态可通过立即返航获得本次 bounded success；
其余成功任务分支不能据此声称复杂 scheduling / RL 必要，也未证实简单规则能满足 5% episode 风险约束。
无新 policy、OracleSafe-SJF、MPC、训练或 98% kill test。等待下一轮审阅。

## 可审计证据与复现

- `integrity.json`：既定 collector 完整性结果。
- `raw_census.tar.gz`：完整运行目录文件，包括输入、plan、root full-state snapshots、分支 traces、
  worker manifests/results/status、最终及保留的上一检查点、启动记录与日志；只排除进程锁文件。
- `raw_inventory.json`：archive 每个 member 的路径、字节数、SHA256。
- `archive_verification.json`：打包后逐 member 读回校验结果。
- `SHA256SUMS`：本目录交付文件校验和。pickle 为归档状态，需要原可信 runtime/依赖，不是通用交换格式。
- `package_results.py`：只读取已完成结果，派生互斥分类并打包，不导入 simulator。

原始 manual collect 命令（aggregate 已存在时不要重复运行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /mnt/workspace/zjl-exp/.venv/bin/python \
  persistent_uav_throughput_v1/scripts/oracle_recoverability_census.py collect \
  --output /mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/artifacts/oracle_recoverability_census_20260920
```

从仓库根目录重建证据包：

```bash
python3 persistent_uav_throughput_v1/evidence/oracle_census_complete_20260920/package_results.py \
  persistent_uav_throughput_v1/artifacts/oracle_recoverability_census_20260920
```

归档 gzip 时间戳可能不同；逐 member 内容哈希是原始证据等价性的依据。
