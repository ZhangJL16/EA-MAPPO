# 冻结 SAC：原生新地图测试启动记录

2026-09-23。协议：[SAC_NATIVE_QUALIFICATION_PROTOCOL.md](SAC_NATIVE_QUALIFICATION_PROTOCOL.md)。模型 SHA-256 `fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be`。源码清单 SHA-256 `458d56c83868846f1f2e5e7e966d8bca25218a135b86a2797adcdc3fe8d6ccbd`。检查点、固定 4 图 × 4 航段清单、两条结果保存在 [`artifacts/sac_native_new_maps_20260923/`](artifacts/sac_native_new_maps_20260923/)；SHA、任务数、不同地图及逐条文件经核验一致。

| 航段 | 地图 | 结果 | 模拟时间 | 统一接触 | HOCBF fallback 步 |
|---|---:|---|---:|---:|---:|
| `m00_p00` | 0 | 到达 | 268.3 s | 0 | 12 |
| `m01_p00` | 1 | 到达 | 87.6 s | 0 | 0 |

当前 2/16 只说明测试路径启动正常。依照 `/home/zjl/mappo/AGENTS.md` 的“After startup health is confirmed, hand the resumable run back to the user”要求，停在首个检查点；没有自动跑完或训练。续跑命令（在本项目目录）：`/home/zjl/mappo/.venv/bin/python sac_native_qualification.py --output artifacts/sac_native_new_maps_20260923 --resume`。

历史固定单图物理 calibration：988/1000 到达、12 次超时、统一接触总数 1，见 [`CALIBRATION_REPORT.md`](/home/zjl/mappo/persistent_uav_throughput_v1/CALIBRATION_REPORT.md)。其样本未并入本次新地图测试。解析导航另在有限高度盒/圆柱的 60×60×24 地图上完成 89/96；地图、动力学上限、到达半径和起终点分布不同，两个百分比不可直接排名。
