# 实施交接：第一批任务，不含训练授权

## 先在本地提供只读清单

把本包放在仓库外，例如 `~/ea_mappo_abc_plan/`。以下脚本不会 import 项目、不会安装依赖、不会运行 pytest/simulator/learner，不读取 artifact 内容，不遍历 CONFIRM。唯一写入是仓库外的新 JSON。

```bash
python3 ~/ea_mappo_abc_plan/local_readonly_inventory.py \
  --repo /home/zjl/mappo \
  --output ~/ea_mappo_local_handoff.json \
  --include-hardware
```

`/home/zjl/mappo` 来自你的历史路径，请在本地确认或替换。输出文件存在时脚本拒绝覆盖；使用新的名字。缺少 nvidia-smi 不影响输出其他清单。检查 JSON 中文件名是否适合分享，再上传；不要提供 SSH key、PAT 或密码。

助手对这个 helper 做了语法编译、帮助入口和临时仓库上的只读行为 smoke；没有在 EA-MAPPO、本地服务器或科学 artifacts 上执行。helper 本身不认证原实验结果。

## 交给本地实现代理的任务范围

> 基于 `master@fff2b026ca3e4ea6b68cacc3964971b0749bfeb3`，先报告 local HEAD 和 dirty 差异，不 reset、不覆盖、不自动 pull。阅读现有 AGENTS、task_plan、notes。遵守更近的用户授权与冻结边界，不把历史计划当新执行命令。
>
> 只实施第一批：新建独立 `research/feedback_protocol/` 工作线；建立 PublicProblem/PrivateTruth、known graph/time/resource/reload/protocol schema；参数化固定 Bernoulli fixture；提供 Bayes finite-budget reference 和 simple public-channel-coverage baseline 的接口；编写必要单元测试。原 bundling 文件和 artifacts 保持不变。
>
> 先明确新实验 terminal contract、Bayes 训练目标及 expected-regret 评价；不声称 minimax guarantee。不把 true-label optimal action 作为未知模型 learner 的 teacher。
>
> 不实现大规模训练、不访问任何旧 CONFIRM、不恢复 T4096 或 UAV、不自动安装到旧 .venv、不启动长实验。新路径/commands 写清楚已实现还是计划。
>
> 交付差异文件清单、依赖闭包、测试定义、数据流/泄漏说明、资源 profiling 方案。运行 focused tests/tiny smoke 必须按用户实际授权；完成后返回审查，不自动升级运行范围。

## 用户需要补充的执行参数

| 项目 | 需要填入 |
|---|---|
| 本地 repo 与 HEAD | inventory 输出 |
| 原 artifacts | 是否存在；是否可提供已访问记录的 hash/只读副本 |
| 可用 CPU/RAM/GPU | 型号、内存上限、并发限制；不是历史推测 |
| 时间与预算 | 每周人时、可用 GPU 小时、是否有工程协作者 |
| 目标投稿 | 目标年份/会议截止日期，由官方信息另行核对 |
| 授权范围 | 只写分支 / focused tests / tiny smoke / pilot training / frozen confirm 中哪些已允许 |
| 外部审查 | 负责 proof audit 与 prior-art review 的研究者 |

## 第一次授权模板

```text
代码起点：<commit>
允许变更：research/feedback_protocol/ 与新说明文档
禁止变更：冻结 core/runner、旧 artifacts、UAV/PSPS/CONFIRM
允许运行：<focused tests / tiny smoke / none>
资源上限：<CPU/RAM/GPU/墙钟>
允许数据分区：<新 debug-only / none>
停止点：<测试报告或首 checkpoint>
是否允许科学分析：<yes/no>
是否允许自动恢复：no
```

该模板不是一次实际授权。详细实施要求与证据边界见主计划。
