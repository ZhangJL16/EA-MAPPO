# 初始首任务下界实验：启动交接

日期：2026-09-29。独立研究项目，历史 PersistentUAVThroughput-v1 只读。当前实验研究原始满电、有三项待办任务的初态，是否存在“所有合法首任务都会在第一次服务选项内耗尽”的事件 `Z`。若某容量档在预定 100 个初态上的同时单侧精确下界证明 `P(Z) > 0.05`，才可推出该档对应 9 个原始 regime 中任意合法高层策略均达不到 5% 耗尽风险目标。未拒绝该条件不能证明存在安全策略。

8 个 worker 的可续跑运行已启动。`runs/initial_choice_lb_v1/manifest.json` 固定源码、模型、regime 和新种子，`rows/` 逐项原子保存结果。`deliverables/startup_health.json` 是首次健康检查快照：已严格校验 85/300 个初态（按三档为 81、4、0）；所有档仍未完成，故没有执行固定样本推断。启动检查无报错，早期结果的 `Z` 计数均为 0；这只说明已见初态存在至少一个不会在首选项内耗尽的选择，不代表完整 `T` 风险或统计结论。

运行中不要再次启动相同分片。状态查看（在本项目目录中，一行命令）：`python3 checks/collect_initial_choice_lb.py --output runs/initial_choice_lb_v1`。只有 3 档各自达到 100 个有效初态后才读其检验字段；未完成档返回 `null`。如果执行会话中断，先确认原 worker 已停止，再用 `MPLCONFIGDIR=/tmp/mpl-uav-feasibility /home/zjl/mappo/.venv/bin/python checks/start_initial_choice_lb.py --workers 8 --wait` 续跑；已验证行会跳过，损坏行会报错而非悄悄覆盖。

前几轮可复检主张及其原始证据边界见 `PRIOR_ROUND_TEST_MATRIX.md`。下一项完整 `T` 的合法固定策略比较尚未启动；不得把本次首选项下界当作它的代用品。
