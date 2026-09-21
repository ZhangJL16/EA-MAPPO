# Infinite-queue intervention v1

准备中：固定 41 个历史 matched divergent pairs，29 roots、70 个唯一 continuation。

[协议](../../INFINITE_QUEUE_INTERVENTION_V1.md)规定只从原始 root 开始解除有限队列容量：容量取整个冻结任务流大小的足够上界，历史丢单不补回。首动作、物理、任务流、充电及 downstream Oracle-Safe SJF 不变。

此实验条件是“原先已经分叉”的配对。比较原/新 signed gap 与 absolute gap，unknown 不记零；不据中间结果调整样本或方法。效果是解除未来有限队列约束的总干预效应，不是唯一中介作用的归因比例。

脚本 `scripts/infinite_queue_intervention.py` 支持续跑；worker 检查点与原始事件本地保留，全部完成后独立 collector 汇总。当前状态见后续 startup_health.json，不将启动证据当作科学结果。
