# B0–B5 diagnostic：B4 validation运行交接

源码提交：b8cb673。29项测试与更新continuing smoke均通过。
旧1000-job calibration和全部27个regime保持不变，未重跑。

当前启动的是B4的阈值validation：27 regimes × 4 thresholds × 10 seeds = 1080 runs。
每run均为固定T的持续任务流，不是单任务calibration。B0–B5 evaluation尚未启动。
启动核验记录见[evidence/v1_3/diagnostic_startup/health.json](evidence/v1_3/diagnostic_startup/health.json)。
完成启动核查后交还，进程继续运行；不监控至完成、不自动启动evaluation。

运行目录：`artifacts/diagnostic_validation_20260920/`。
日志：`artifacts/diagnostic_validation_20260920.log`。
PID与完整启动命令：`evidence/v1_3/diagnostic_launch.json`。

暂停：确认PID仍对应此命令后发送`kill -TERM 252399`，当前step后原子保存并退出。
不要用SIGKILL。续跑使用：

```bash
cd /home/zjl/mappo/persistent_uav_throughput_v1
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli baselines \
  --frozen evidence/v1_2/calibration/frozen_regimes.json \
  --calibration-compatibility evidence/v1_3/calibration_compatibility.json \
  --output artifacts/diagnostic_validation_20260920 \
  --split validation --regimes all --checkpoint-policy-steps 100 --resume
```

运行中不要修改`persistent_uav/*.py`或冻结依赖，源码变化会被resume校验拒绝。
validation全部完成才生成`thresholds.json`；无经验可行阈值的regime标记B4不可用，
不删除失败run、不换seed或阈值。10个seeds不构成5%风险保证。

解释结果前仍需estimator误差审计；第一轮结果只作结构诊断。
最终98% kill test须等待MPC、small reference与estimation/planning decomposition。
神经训练禁止，MPC/Oracle暂缓，本次均未执行。
