# 复现与证据索引

## Material Passport

本轮对象：PersistentUAVThroughput-v1.2，单地点空间服务，不是pickup–delivery。
源码提交：`15fe98e14a84e109964d0b5570f53d024a8e54d3`。
原v1.1源码：`1a13bf0`，完整Git bundle随证据保留。
旧导航依赖基线：`d8c3a6471eb5ecfd00004bc3805251ccfc78cab2`。
当前仓库直接跟踪新代码和精简证据，不是嵌套仓库/submodule。

## 从哪里开始审

| 问题 | 证据 |
|---|---|
| 新实现是不是只有文字描述 | `persistent_uav/`、`tests/`及源码提交 |
| 对应哪个物理代码版本 | `evidence/v1_2/source_revision.json`：逐文件manifest SHA对源码commit核对 |
| 哪个SAC/地图/输入pair | calibration manifest；固定1000 pair及地图均在其中 |
| 原来的16项检查是否真实存在 | `evidence/original_v1_1/tests.txt` |
| 修复后检查 | `evidence/v1_2/tests.txt`，22项逐项结果 |
| 真正continuing了吗 | `evidence/v1_2/continuing_smoke/continuing_trace.json` |
| 恢复后outcome是否一致 | 同目录 `recovery_equivalence.json` 与 `midflight/` 检查点 |
| calibration为什么中断过 | `evidence/original_v1_1/error.json`：地图生成保护距离拒绝合法起点 |
| 是否换图/删除难样本 | repaired migration记录及最终integrity：同一1000 pair、同一地图、原完成前缀保留 |
| 是否训练 | no trainer/optimizer invocation；actor inference-only；manifest与status均记录neural updates=0 |
| 是否跑baseline或oracle | baseline入口明确暂停；研究运行只有physical calibration |

原中断的status仍写running，这是旧错误处理的缺陷，不代表当前仍有原进程存活。
原始文件不改写，error记录及修复说明优先。修复后错误处理会将status标为error。
旧的1秒smoke只覆盖短飞行；新的continuing trace覆盖serve→serve→recharge→serve全过程。

## 本地复现检查

在 `persistent_uav_throughput_v1/` 执行：

```bash
/home/zjl/mappo/.venv/bin/python -m unittest discover -s tests -v
/home/zjl/mappo/.venv/bin/python -m persistent_uav.continuing_smoke --output artifacts/new_continuing_smoke
```

新smoke输出目录必须为空。这是工程复现，不是正式baseline。
现有工程fixture的B、r、T不参与27-regime参数冻结。

## 执行资产与可迁移性

`evidence/v1_2/runtime_assets.json`给出精确SAC ZIP、native solver C源码/二进制及打包文件哈希。
本地 `artifacts/runtime_assets.tar.gz`另供下载，避免把26MB模型重复塞入Git源码历史。
native solver的补充记录是在修复运行期间捕获，并非原startup manifest已有字段。
平台、Python及关键依赖版本分别在source_revision与calibration provenance中。

若解压完整复现快照到其他目录，应设置 `PERSISTENT_UAV_LEGACY_ROOT` 为解压后的旧导航源码根目录。
导入器支持该变量。manifest保留采集时绝对路径作为证据；不得为了让resume通过而直接改SHA。
异地重跑应创建新输出目录；旧检查点跨路径/版本使用需显式记录迁移。
提供的native二进制对应本次Linux x86_64环境，不保证跨平台逐位一致。

## 统计解释边界

导航成功率条件于一张固定地图与预先生成的IID x→y分布，不是跨地图可靠性保证。
成功任务的时间/能耗分位数明确是条件统计；timeout保留实际消耗，不作为成功总代价。
随机pair分布也不完全等于scheduler诱导的转移分布；站点往返的充分可靠性尚不能由此断言。
障碍暴露是离线直线段几何诊断，不能声称已记录实际路径上所有LiDAR暴露。
所有Wilson区间均为pointwise区间，不是多分组同时置信保证，也不是5% depletion风险证书。
资格筛查是在pilot已启动后、此次读取结果前写定；诚实标为pre-analysis，不冒称采集前预注册。

calibration完成后只做navigation qualification和参数核验，仍等用户批准B0–B5。
