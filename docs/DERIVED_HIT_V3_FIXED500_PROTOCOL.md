# v3 派生命中通道：固定500任务评测

用户在完整训练结束后明确同意进行正式评测。冻结最终524288步checkpoint，
不训练、不挑选最好中间checkpoint、不自动启动后续实验。

## 与旧对照保持一致

使用原始五档各100任务：100–500、500–1500、1500–2500、2500–4000、>4000米。
文件SHA256为6169ef6ec56e38d9c95260a1351bd40cfc8af0e62b6672dac93013494e00692d；
seed170001+原始任务下标，同一起点/初速度/目标、24障碍、4000政策步期限。
仍为确定性SAC原始动作，不使用HOCBF。碰撞修复、速度清零、重复碰撞固定折扣、
统一政策步计数均不改变。外部1039维观测仍与v2相同，仅模型内部派生命中通道。

模型：artifacts/deployable_derived_hit_v3_sac_20260907_v1/sac/checkpoint_000524288/model.zip。
SHA256：372e70ffc51d134c0b28dd17012f579a6be5f69e914c71943523bef9a1cab194。
独立入口scripts/evaluate_deployable_derived_hit_v3.py，仅增加v3版本校验与清楚的
协议标签。停车wrapper、任务采样、推理、指标、完整组保存与恢复函数的AST
与v2入口一致；复用已通过的恢复证据。一次最终模型5任务×2步GPU smoke通过，
显式formal=false，不据此判断表现，不添加性能gate。

## 结果与恢复

正式输出：artifacts/deployable_derived_hit_v3_fixed500_20260907_v1。
8并行worker，按完整组原子保存evaluation_stratified.json；中断后同命令加
--resume，只重跑尚未提交的组。SIGTERM/SIGINT或输出目录PAUSE请求可暂停。
训练产物和旧v2/旧SAC评测不覆盖。协议绑定模型、任务和源码hash。

```bash
uv run --no-project --python .venv/bin/python python scripts/evaluate_deployable_derived_hit_v3.py \
  --source artifacts/deployable_derived_hit_v3_sac_20260907_v1 \
  --output-dir artifacts/deployable_derived_hit_v3_fixed500_20260907_v1
```

完成后应按source_task_index与v2和旧SAC配对，分别报告到达与零碰撞到达、
统一碰撞尾部、超时，以及共同到达任务的路径/能耗。只有一个训练seed，
不能作跨seed保证；固定任务多次使用，不能叫全新holdout。该评测不包含
预算约束、主动返航或持续充电服务，不能声称能量可持续性已解决。

只验证首个提交组后交接，等待用户查结果；不监控完成或自动追加训练。

交接记录：Python PID20663；启动检查时200/500已提交，原始任务前缀与统一
安全到达语义通过校验，正式v3合同确认，无ERROR，进程存活。此后停止检查。
