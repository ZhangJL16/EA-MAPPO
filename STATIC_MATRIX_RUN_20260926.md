# 二维静态全图研究矩阵：运行记录

2026-09-26 启动于独立项目 `/home/zjl/uav_learning_research`。旧仓库 `/home/zjl/mappo` 只提供已安装的 Python/SB3 依赖解释器，未写入旧研究代码或历史结果。

主问题是：在静态已知全图、确定性能耗模型和同一联合安全过滤器下，学习策略能否在未见地图的 600 仿真秒内，比验证集预选的有限视界 MPC 完成更多目标。强 MPC 若达到或超过 PPO，本单目标主张如实判失败；多目标队列属于另一个后续问题。

固定划分：训练图 8–31，验证图 32–39，测试图 40–55；PPO 种子 101/202/303，每个 51,200 个高层决策。验证和测试均运行路线补满、路线部分补能、MPC 深度 1/2，以及三个 PPO 种子。所有结果必须同时报告碰撞、耗尽、返站失败、补能、过滤器干预及规划计算。

主运行目录：`artifacts/dual_constraint_2d_static_matrix_v1_20260926/`。其中 `manifest.json` 锁定源码与依赖哈希，`status.json` 显示阶段，各子任务有原始事件日志、状态与检查点。可用下列**单行**命令恢复同一运行：

`/home/zjl/mappo/.venv/bin/python -m dual_constraint_2d.matrix_runner --output artifacts/dual_constraint_2d_static_matrix_v1_20260926 --workers 3`

首次启动目录 `artifacts/dual_constraint_2d_static_matrix_20260926/` 在第一检查点前遇到训练探索位置超出规划器额外净空的异常，保持原样作失败证据。修复后的 512 决策 smoke 位于 `artifacts/dual_constraint_2d_matrix_repair_smoke_20260926/`，成功保存 `model_000000512.zip`。正式 v1 目录单独建立，不混合旧尝试。

当前为运行中，不填写尚未产生的比较结果。只有 `analysis.json` 和 `RESULT.md` 写出后，才能解读完整静态矩阵。

启动健康核查：三个训练种子均已写出各自 `model_000005120.zip` 和 `status.json`，矩阵调度进程仍在运行；这证实首个正式检查点和中断恢复入口可用。训练 episode 中的碰撞与耗尽属于允许的试错记录，不能解释为正式评价安全性。
