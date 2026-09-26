# GPU-MPC 速度审计与并行 v3

日期：2026-09-26。研究项目：`/home/zjl/uav_learning_research`。
本记录不把 GPU 使用率等同于实验质量，也不改变碰撞、电量或奖励语义。

## 实测瓶颈

在 v1 留存的 H4/map 32 未完成断点上，固定同一状态重复 5 次 `act()`：

| 部分 | 总墙钟秒 | 占本次决策墙钟比例 |
|---|---:|---:|
| GPU 候选代理评分，含 Python 调度与同步 | 0.754 | 26.6% |
| CPU 精确安全证书 | 2.084 | 73.4% |
| 合计 | 2.839 | 100% |

5 次决策评分 1,620 个 GPU 候选，只精确核验 5 个 CPU 候选。
`cProfile` 显示 564 万次 Python 调用；几何障碍检查、路线规划、
CBF/OSQP 与返站 rollout 是主要 CPU 工作。这个冻结状态是局部画像，
不能据此推断所有地图的精确比例。运行时 `nvidia-smi dmon` 的原串行阶段
10 次 SM 样本约 9%–23%；新并行阶段 10 次平均 10.4%，说明仍然是
CPU 安全层和大量小 GPU 操作交替造成的低利用率。

PyTorch 官方 `torch.compile(mode="reduce-overhead")` 针对小批次 Python/
CUDA launch 开销，但只优化当前约四分之一的代理评分部分，并非精确安全层：
<https://docs.pytorch.org/docs/stable/generated/torch.compile>。
NVIDIA Warp 的逐候选 GPU kernel 与 graph 可作为后续重写仿真/几何部分的参考，
但必须重新核验与当前接触恢复、统一碰撞计数、返站证书的等价性：
<https://github.com/NVIDIA/warp/blob/main/docs/user_guide/runtime.rst>。

## 已实施的加速

原 `gpu2d.campaign` 串行执行不同地图/H4/H8 作业。
新 `gpu_parallel_campaign.py` 只把**独立作业**交给 3 个进程并行；
每个作业仍调用未改动的 `gpu2d.evaluate`、原代理评分、
原 CPU 精确证书和原环境。新入口放在源码哈希集合外，
但自己的 SHA 和算法源码 SHA 都写入 v3 manifest。

固定地图 36/37/38，各 20 决策的独立基准：

| 调度 | 墙钟秒 |
|---|---:|
| 串行 | 33.547 |
| 三作业并行 | 13.790 |

速度比 2.433。三张图的 `events.jsonl` 在串行和并行两种执行方式下
逐字节相同。测量时旧 v2 任务在后台，因此这是有干扰的探索性吞吐基准，
不是完整 48 作业的最终加速比。原始数据在
`profile_output/parallel_benchmark.json`。

v2 运行于已保存检查点处被显式中止，并留下中止错误记录。
v3 复制了当时 3 个已有作业（其中 2 个完成、1 个部分完成），
逐文件 SHA 记录在 `artifacts/gpu_mpc_parallel_v3_20260926/seed_migration.json`。
v3 从 2026-09-26 12:33:45 CST 起运行，原定截止 20:33:45 CST，
计划 48 作业、3 个并行 worker。启动时 H4/map 33、H8/map 33、
H4/map 34 都保存了新检查点且无错误，H4/map 33 保留 v2 事件前缀。
**用户随后要求先停止运行，验证速度与内存后再跑。**v3 已停止，
5/48 作业完成、7 个检查点保留；`user_stop_record.json` 保存停止原因，
原 `status.json` 另存为 `status.before_user_stop.json`。当前无长实验运行。

## 扩展并发与 24 GB RAM 约束

8 张验证图每图 20 决策的短测：1/2/4/8 workers 的墙钟分别为
90.36/46.42/26.74/14.90 秒；8 workers 比串行快 6.06 倍。
16 张训练图每图 20 决策：8/12/16 workers 的墙钟分别为
30.27/29.42/19.66 秒。所有 worker 配置的对应事件日志逐字节相同。
这些数字只测短任务吞吐，不是完整 48 作业的最终加速比。

内存采样：

| Workers | 进程独占内存峰值 MiB | 进程 RSS 总和峰值 MiB | 系统最低可用 RAM MiB | 显存使用峰值 MiB | Swap MiB |
|---:|---:|---:|---:|---:|---:|
| 8 | 6477 | 8700 | 14845 | 2261 | 0 |
| 16 | 12950 | 17462 | 8346 | 3279 | 0 |

物理 RAM 约 23.47 GiB。16 workers 在空闲机器的短测没有 OOM，
但会占用较多 RAM；正式运行默认**8 workers**。
新 `gpu_parallel_campaign_v4.py` 在 RAM <32 GiB 时禁止超过 8 workers，
且 8-worker 启动前要求 `MemAvailable >=12 GiB`。验证短测与断点恢复后，
v4 已按用户条件启动；首检查点与资源状态见
`GPU_MPC_V4_MEMORY_SAFE_STARTUP_20260926.md`。内存原始数据：
`profile_output/memory_benchmark_8.json`、`memory_benchmark_16.json`。

另一组隔离 v4 源码候选把同位置 9 次路线查询合为一次，并为圆柱线段检查
增加保守包围盒快速排除。3 个 map 的候选分数/动作相同；约 4.8 万次线段
对照相同，20 决策完整事件日志相同。但一条轨迹端到端仅从 13.42 秒降到
12.29 秒，暂不写入正式实验源码。候选补丁与短测结果保存在
`profile_output/v4_candidate_only.patch` 和相应 JSON。

## 尚未解决的 GPU 利用率

3 个进程提高的是**每小时完成的地图数**，并没有把安全证书搬到 GPU。
要进一步提高 GPU 计算占比，需要把多条候选和多张图的几何、动力学、
CBF 与返站证书组成批量 kernel，或先显著压缩 CPU 几何/路线重复计算。
这样的 v4 必须新立源码版本，保留 v3 的原始证据，并在大量状态上
逐步比对原实现的动作、能耗、接触恢复与返站判定。不能用近似代理
评分替代评价期的精确安全证书。

## Profiling instrumentation changelog

| 文件 | 变更 | 用途 |
|---|---|---|
| `profile_output/profile_gpu_mpc_v2.py` | 新建 | 只读加载旧断点并测量 CPU/GPU 阶段 |
| `profile_output/gpu_mpc_v2.prof` | 新建 | `cProfile` 原始数据 |
| `profile_output/gpu_mpc_v2_cpu_top.txt` | 新建 | CPU 热点摘要 |
| `profile_output/gpu_mpc_v2_torch_ops.txt` | 新建 | PyTorch CPU 操作统计；本环境 CUPTI 无 CUDA timeline |
| `profile_output/gpu_mpc_v2_profile.json` | 新建 | 阶段墙钟与候选计数 |
| `profile_output/benchmark_parallel_jobs.py` | 新建 | 独立作业串并行对照 |
| `profile_output/parallel_bench/` | 新建 | 6 条 20 决策测试轨迹 |
| `profile_output/parallel_benchmark.json` | 新建 | 吞吐与事件哈希 |
| `profile_output/gpu_parallel_dmon.txt` | 新建 | 并行阶段 10 次 GPU 采样 |

未为 profiling 修改 `dual_constraint_2d/`、`gpu2d/` 或 `nav3d/`
中的实验源码。新建的 `gpu_parallel_campaign.py` 是正式 v3 调度入口，
不是性能测量探针。
