# Ascend 服务器资源记录

记录日期：2026-09-26。来源：用户在 `developer@e9b6249104e74d4a9af4d98ce10ae8ec:/mnt/workspace/MseLoss` 提供的终端输出。原输出没有采集时分秒；内存、NPU 占用和进程信息仅为当时快照，规划作业前须重新检查。

| 项目 | 记录值 |
| --- | --- |
| 系统 | Ubuntu 24.04.3 LTS (Noble Numbat) |
| 内核 | `5.10.0-182.0.0.95.r2220_156.hce2.aarch64`，2024-09-14 构建 |
| 架构 | `aarch64` |
| CPU | 40 个在线逻辑 CPU（0–39）；1 socket，40 cores/socket，1 thread/core；`lscpu` 未报告具体型号 |
| NUMA | 输出仅列出 node 0，CPU 0–39 |
| 内存快照 | 总计 229 GiB；已用 33 GiB；空闲 118 GiB；缓存 78 GiB；可用 195 GiB |
| Swap 快照 | 0 B（未配置） |
| NPU 管理工具 | `npu-smi 25.5.5` |

`npu-smi` 报告逻辑 NPU 3 的两个 Ascend 910 chip。不要据此推断有两张独立可分配的加速卡。

| Chip | Phy-ID | PCI Bus-ID | 健康 | AICore 快照 | HBM 快照 | 已列进程 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 0 | 6 | `0000:0B:00.0` | OK | 100% | 57,425 / 65,536 MiB | PID 1449814，`VLLMWorker_TP`，54,355 MiB |
| 1 | 7 | `0000:0A:00.0` | OK | 100% | 57,163 / 65,536 MiB | PID 1449816，`VLLMWorker_TP`，54,355 MiB |

当时两个 chip 都在运行 vLLM，不能把 64 GiB/chip 的标称 HBM 当作可立即使用的容量。提供的 `lspci -nn | grep -Ei 'Huawei|Ascend'` 输出没有匹配行；NPU 识别以上述 `npu-smi` 为准。

| 设备 | 大小 | 类型 | 型号/挂载点（按输出） |
| --- | ---: | --- | --- |
| `sda` | 200 G | disk | VBS fileIO；`/mnt/workspace` |
| `sdb` | 200 G | disk | VBS fileIO；`/home` |
| `vda` | 500 G | disk | `vda1` 1 G；`vda2` 499 G 挂载于 `/usr/local/Ascend/driver/lib64/driver` |
| `vdb` | 500 G | disk | 未显示挂载点；不能据此假设可用 |

后续实验规划：这台机器适合优先评估 CPU/大内存的独立地图并行任务。用户说明，未来需要在这台服务器跑实验时，交接方式是先将代码提交到 GitHub 仓库；服务器侧有人专门负责将代码适配为 ARM64 可运行版本。因此本地应提供可复现的源码、依赖说明、实验协议及检查点/结果格式，并在交接时明确对应 Git 提交。当前 `gpu2d` 的 GPU 路径使用 CUDA；ARM64 CPU 适配本身不等于 Ascend NPU 适配，后者若需要使用仍须单独确认。安排新作业前重新检查 CPU/内存/磁盘/NPU 占用；此记录不包含可连接的 SSH 地址或端口。
