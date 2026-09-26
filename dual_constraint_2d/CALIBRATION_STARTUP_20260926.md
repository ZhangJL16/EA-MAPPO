# 合成二维标定首检查点

日期：2026-09-26。标定协议见[`SYNTHETIC_PROTOCOL_20260926.md`](SYNTHETIC_PROTOCOL_20260926.md)。这不是训练或正式方法比较。

已实现固定高度二维绕障路由、低速到达参考控制，以及固定 8 张地图 × 每图 8 个政策无关候选目标，共 64 个往返任务的可恢复标定。全部结果完成后才会把电池容量定为成功参考往返能耗中位数的 2 倍，充满耗时定为成功参考往返时间中位数。当前尚未得到这些数值，也尚未实现满电目标筛选或逐步返站安全证书。

7 项二维相关测试通过，包括第一次写入与中断续跑。第一条正式格式的标定结果 `m000_c000` 已写入：站→目标→站成功，总仿真飞行时间 45.75 s、合成能量 65.8595，统一接触 0、QP 不可行步 0。当前 **1/64**；没有启动后台任务，也没有监控到完成。

原始记录：[manifest](../artifacts/dual_constraint_2d_calibration_20260926/manifest.json)、[首条结果](../artifacts/dual_constraint_2d_calibration_20260926/results/m000_c000.json)。SHA256 分别为 `2c8368bfc733ae8d38a7ed7244e1e50ce3898418facbc3af72ae40dd8a5177d0` 和 `4b7e550696837d1e6c058a12799cb0aab76aade74004275f6b95432fe7e1bb83`。manifest 记录四份执行源码 SHA256；若源码改变，运行器会拒绝混合新旧结果。

续跑使用一行命令：

`cd /home/zjl/uav_learning_research && .venv/bin/python -m dual_constraint_2d.calibration`

续跑只补缺失行。全部完成后产生 `calibration.json`；在此之前不能称电池或充电参数已冻结。
