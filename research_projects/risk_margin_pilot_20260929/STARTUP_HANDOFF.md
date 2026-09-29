# 状态相关余量探索实验：启动交接

2026-09-29 17:44 Asia/Shanghai。用户已授权开启实验。本实验按 [`PROTOCOL.md`](PROTOCOL.md) 固定 3 个 regime × 10 条新流 × 5 方法 = 150 条完整原始窗口回放。原 `persistent_uav_feasibility_20260929` 的 810 条筛查未修改，两个实验目录和 seeds 分离。本轮没有训练，也没有调整冻结物理、导航和碰撞机制。

聚焦策略测试 3/3 通过，新的决策路径完成一次 10 秒、50 个低层 policy steps 的真实导航 smoke。Manifest 初始化后启动**一个 CPU worker**。首次 checkpoint `job_0000` 已原子保存；`collect.py` 完整校验为 `validated=1/150, complete=false`，状态文件显示 `stopping=false`。首行属于 `regime=4, seed=1402609290, method=U0`；只用于检验保存/恢复路径健康，不解读耗尽或完成数。启动时统一控制台 session ID 为 `85703`；该 ID 是本次工具会话标识，不是可跨环境复用的进程号。

完整性指纹：manifest SHA256 `346fb41e1fae5ebc8c766fecb1550e9616a860927b12c838edb9aece12e5a49b`；首行 SHA256 `2ecbd178f9c86ddeda35d085a10b33a5e43c27e07afc7e5442e501a3ba7b597f`。其后运行会继续写 `runs/margin_pilot_v1/rows/`，`collect.py` 在未完成全部预定行之前只给完整性计数，不给方法胜负。

单 worker 有 `worker.lock` 独占锁，**运行中不要再次启动相同输出目录**。如进程中断，先确认旧进程退出，再在本项目目录用原环境执行：

```bash
MPLCONFIGDIR=/tmp/mpl-uav-margin /home/zjl/mappo/.venv/bin/python run.py
```

只读检查当前已保存行：

```bash
MPLCONFIGDIR=/tmp/mpl-uav-margin /home/zjl/mappo/.venv/bin/python collect.py
```

本轮在首 checkpoint 健康核验后交还运行；不监控到完成，不自动进入 27 格、rollout、训练或最终 `δ=0.05` 认证。
