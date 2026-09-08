# Spotlight 执行衔接：分工、关键路径与门禁（2026-09-06）

日期：2026-09-06。状态：用户已确认目标为 **ICML 2027 Spotlight 级论文（4 个主定理 + 多域多基线实验）**。
本文档衔接 codex 正在执行的导航平台工作与返航决策主线，明确先后顺序、止损条件与分工，避免重复实验与口径冲突。

## 1. 决策记录（用户拍板）

1. 投稿：跳过 ICLR 2027，主攻 ICML 2027（预计 2027-01 中下旬截稿）。
2. Gate 拆分：`docs/PLATFORM_GATE_SPLIT_20260906.md`（平台 gate 放行下游；部署 gate 降为工程目标）。
3. 目标形态：Spotlight = 理论定理支撑 + 较多实验；参照 *Fitted Distributional Evaluation*（NeurIPS 2025 Spotlight）、SDAC（NeurIPS 2023）。
4. 论文级创新定位（与 codex 台账一致）：**能量/返航可行性**，绝不是导航基线选型。证据包清单见 `independent-review-20260906/REVIEW_AND_ROADMAP.md` §7。

## 2. 当前 codex 工作与本路径的关系（不冲突、不打断）

- 正在跑：`recovery_sac_ppo_scratch_20260906_v1`（SAC 臂 524288 步，然后 PPO 臂）→ 属于 **G0 平台决策** 的输入之一。
- 已关闭：critic completion（正反流不一致）、PPO 更新节奏修复（已并入 scratch 协议）——不再重启这些分支。
- 规则：**scratch 对跑完后不再开任何新导航变体**；无论结果如何，进入 G0 止损判定（见下），并把全部资源转回主线。

## 3. 关键路径（按序执行，每步有门禁）

| Gate | 内容 | 通过条件 | 现有资产 |
|---|---|---|---|
| G0 平台 | 导航平台定案 | 二选一：(a) scratch SAC/PPO 任一臂 shield-off 开发集 50 任务到达 ≥60% → 以其为候选平台；(b) 均 <60% → 冻结 R3+HOCBF（shield-on 47/50）为平台。随后按 PLATFORM_GATE_SPLIT §2 跑 500+50 平台 gate（≥0.85 放行） | 方向保留编码器、locked recovery 环境、scratch runner |
| G1 恢复策略修复 | 返航可靠性 | 310 锚点复测返航成功率 ≥97%（当前 88.71%，失败模式：17 超时/16 碰撞/2 边界） | `dual_viability_counterfactual_diagnostic_150scenes_20260905_v1`、`run_dual_viability_counterfactual_diagnostic.py` |
| G2 随机性注入 | 概率对象成立 | 风/执行噪声/动态障碍（固定 seed 分布）后 Var(Z_C)>0 且可复现；重跑概率语义审计 | `audit_return_probability_semantics.py` |
| G3 Oracle headroom | 决策效用存在 | 5 点×320 循环；Oracle 对 SOC/距离 throughput 增益 ≥5% 且 stranding 不劣化；<5% 停 learned-manager 线 | `run_return_decision_stage_b.py`（需修 4000 步 stall 与 seed 级原子写） |
| G4 pair-vs-interface 机制 | A 线生死 | 2 策略×2 滤波器 leave-one-pair-out，oracle 误差下"interface extrapolation 解释力 > pair identity"（预注册统计） | Stage E/F 协议、R3 冻结平台 + 新训练一个策略 |
| G5 理论加锋 | Spotlight 分水岭 | **T2 不可辨识性分离构例**（普通 support 重叠、尾部/目标 support 不重叠）成立；T1 充要条件、T3 传输界、T4 边界稳定性收敛为 4 主定理（其余 30 条进附录） | `RETURN_TO_CHARGE_DERIVATION_PACKAGE.md` Steps 1-6、Theorem 27 Assouad 构造 |
| G6 预测器套件 | 方法按需 | 直接 MC/TD、PCM、executed-WM、WM+ensemble；AURC/风险-覆盖 gate：ensemble 打不过才启用 SIRP | Stage C 协议 |
| G7 多域复制 | Spotlight 实验广度 | 第二域（2D 地面车燃料域）+ 第三域（2D grid 资源世界）+ 第二滤波器族（MPC/规则 shield）复现 G4 机制 | 新建轻量环境 |
| G8 收口 | 投稿包 | ≥3 seeds headline + CI；配对 stranding–throughput Pareto；消融（occupancy 加权、截断、abstention、适应预算）；可复现包 + 独立证明审计 | ICLR 稿骨架 |

## 4. 分工与冲突规则

- **codex（执行侧）**：G0–G4、G6–G8 的实现与实验（沿用现有 fail-closed、预注册、原子产物纪律）。
- **独立审阅（本目录）**：只做审阅/文献/协议，不启动实验、不修改 codex 运行中的 artifacts；G5 的理论骨架可与 codex 并行推进（纯数学，不占 GPU）。
- 口径统一：论文 novelty 语句一律用 §1.4 的定位；"自适应返航方法"不得作为 headline（文献已占位），headline 是"executed-interface 上资源律的可辨识性 + 返航决策的轨迹级统计可靠性"。
- 时间盒：G0 最迟 2026-09-20 定案；G3 最迟 2026-10-15 出数；G5 最迟 2026-11-30 出分离构例（做不出则按 poster 级包装，不硬冲 spotlight）。

## 5. 与既有文档的关系

- 本文件是 `PLATFORM_GATE_SPLIT_20260906.md`（gate 数值与止损）与 `independent-review-20260906/REVIEW_AND_ROADMAP.md` §7（spotlight 证据包）的执行衔接层。
- 与 `RETURN_TO_CHARGE_ICLR_RESEARCH_PROTOCOL.md` 的 Stage 编号一致；其 §12 evidence status 继续作为状态台账。
- codex 的 `RECOVERY_SAC_PPO_DECISION_LEDGER.md` 继续作为 G0 平台的决策依据，不重复其内容。
