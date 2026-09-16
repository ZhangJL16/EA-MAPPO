# Paper Card: Sumita & Rieders (1988)

> Source coverage: Abstract only, plus publisher metadata and reference list
> Extraction confidence: Mixed
> Locator mode: source-limited
> Primary analytical lens: Methods and theorem-scope boundary
> Secondary analytical lens: Prior-art threat to recurrence-tail abstraction
> Context verification: Targeted external check
> Card completeness: Partial

The full article was not downloaded or inspected. Statements about theorem
hypotheses, proofs, and scope beyond the abstract are therefore marked
unavailable rather than inferred.

## 01 基本信息

| 字段 | 内容 |
|---|---|
| 标题 | *First passage times and lumpability of semi-Markov processes* |
| 作者 | Ushio Sumita; Maria Rieders |
| 机构 | University of Rochester（出版社页面所列） |
| 期刊 | *Journal of Applied Probability*, 25(4), 675--687 |
| 年份 | 1988 |
| 类型 | 理论研究论文 |
| 领域 | semi-Markov processes; lumpability; first-exit/first-passage times |
| DOI | 10.2307/3214288（出版社页面）；页面亦链接 DOI 记录 |
| 代码/数据 | 不适用 |
| 阅读日期 | 2026-09-16 |
| 在本项目中的位置 | 对“return-time abstraction 是否只是 semi-Markov lumpability”的最高优先级先验工作 |
| 主来源 | [Cambridge Core publisher page](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/first-passage-times-and-lumpability-of-semimarkov-processes/F5601F9E5DF2EC1C3259DDE88907E665) |

## 02 一句话总结

论文在 Laplace-transform 域中，把 Serfozo 的 semi-Markov lumpability
充要条件重新解释为 first-exit-time 条件，并进一步通过 first-passage
times 建立一个新的充要刻画；由于本轮仅获得摘要，具体定理的状态空间、
初始分布和量词范围尚不可核验。[Paper: Abstract]

## 03 研究问题

- 具体问题：一个 semi-Markov process 的状态聚合何时仍产生合法的
  lumped semi-Markov process？
- 重要性：lumpability 决定能否在不丢失目标动力学结构的情况下压缩状态空间。
- 摘要所述不足：既有 Serfozo 条件需要用 first-exit-time 重新解释，并希望
  建立 first-passage-time 与 lumpability 的直接关系。
- 精确问题：**Can semi-Markov lumpability be characterized necessarily and
  sufficiently through first-exit and first-passage-time transforms?**

[Paper: Abstract]

## 04 研究背景与发展路径

| 阶段 | 代表工作/思想 | 本文位置 | 证据状态 |
|---|---|---|---|
| Markov 函数与状态聚合 | Burke--Rosenblatt、Dynkin、Kemeny--Snell 等 | 参考文献中的基础 | 出版社参考文献可见 |
| Semi-Markov 函数 | Serfozo (1971) | 本文重新解释其充要条件 | 摘要明确陈述 |
| First-exit / first-passage characterization | Sumita--Rieders (1988) | 本文核心 | 摘要明确陈述 |
| Passage-time-preserving aggregation | Bradley (2002) 及后续计算工作 | 后续外部连接 | 外部来源核验 |

该发展路径只有“本文自述位置”和参考文献层面得到核验；没有全文，不能恢复
作者在正文中如何评价其他路线。

## 05 论文识别的核心痛点

| 痛点 | 表现 | 原因或作者解释 | 论文证据 |
|---|---|---|---|
| Lumpability 的可判定刻画 | 需要判断聚合后过程是否仍为 semi-Markov | 摘要表明已有 Serfozo 充要条件可被 first-exit 重新解释 | Publisher abstract |
| Passage quantities 与聚合条件的关系 | first-passage 信息能否刻画 lumpability | 作者声称建立新的必要充分条件 | Publisher abstract |

摘要未提供计算复杂度、适用状态空间或可学习性痛点。

## 06 核心思想

1. **表层方法：**在 Laplace-transform 域研究 first-exit/first-passage
   transforms 与 lumpability。
2. **核心洞见：**semi-Markov 状态聚合的结构条件可以用 passage-time
   对象做必要充分刻画，而不只是写成局部转移核条件。
3. **[Analysis] 可迁移教训：**任何以“return-time law 刻画
   semi-Markov abstraction”为核心的新工作，都必须先排除只是该经典刻画的
   受控或近似改写。

前两项来自 [Paper: Abstract]；第三项为本卡分析。

## 07 方法概览

- 输入：semi-Markov process、状态空间 partition，以及相关 first-exit / first-passage 对象。
- 输出：lumpability 的必要充分条件。
- 数学工具：Laplace transforms。
- 训练、数据和反馈环：不适用。
- 具体假设：全文不可得，无法可靠列出。

[Paper: Abstract]

流程（摘要可支持的最小版本）：

```text
semi-Markov process + partition
        -> passage-time transforms
        -> necessary-and-sufficient conditions
        -> lumpability decision
```

## 08 核心模块拆解

| 模块 | 功能 | 必要性 | 输入与输出 | 支持证据 | 移除后的影响 |
|---|---|---|---|---|---|
| First-exit reinterpretation | 重述 Serfozo 条件 | 连接既有 lumpability 理论与 passage quantities | partition/process -> first-exit condition | [Paper: Abstract] | 无法得到摘要宣称的重解释 |
| First-passage characterization | 建立新的充要条件 | 将 passage time 与 lumpability 直接关联 | first-passage transforms -> lumpability | [Paper: Abstract] | 失去论文主要新增结论 |
| Laplace-transform analysis | 提供推导域 | 摘要称方法完全基于该域 | time laws -> transforms | [Paper: Abstract] | 证明路线不可恢复；具体影响全文不可评估 |

## 09 必要公式与符号

全文未访问，不能从摘要可靠重建作者公式。以下仅为概念占位，不归于作者原式：

\[
  \text{partition lumpable}
  \quad\Longleftrightarrow\quad
  \text{specified first-passage/first-exit transform conditions}.
\]

作者的 transform 定义、矩阵公式、状态空间条件和定理编号均为
**Not assessable from the supplied source**。

## 10 实验设计与证据链

这是理论论文；摘要未报告实验、数据集、仿真或消融。

| 实验/证明 | 所检验主张 | 条件 | 结果 | 可支持结论 | 不可支持的更强结论 | 来源 |
|---|---|---|---|---|---|---|
| Serfozo 条件重解释 | lumpability 可由 first-exit time 表达 | 具体条件不可见 | 摘要声称必要充分 | 存在这种重解释 | 其对 controlled/POMDP 情形直接成立 | [Paper: Abstract] |
| 新的 first-passage 条件 | first-passage 与 lumpability 有充要关系 | 具体条件不可见 | 摘要声称建立 | 存在新的充要刻画 | 任意 return-law equivalence 都等同 lumpability | [Paper: Abstract] |

## 11 对结论的正确解释

- 任务边界：经典 semi-Markov lumpability，不是已确认的 controlled POMDP
  history representation theorem。
- 初始分布量词：不可评估。
- 是否覆盖 weak/strong lumpability：摘要只写 lumpability；不可自行指定。
- 是否覆盖 approximate aggregation：摘要未说明。
- 是否覆盖策略：摘要未说明 control 或 policies。
- 是否覆盖可递归学习表示和有限样本：未见证据。

**边界化重述：**从可访问的出版者摘要只能确认：作者在 transform 域建立了
semi-Markov lumpability 与 first-exit/first-passage times 之间的必要充分联系；
不能确认它是否直接覆盖当前研究提出的 policy-uniform history abstraction。
[Paper: Abstract]

## 12 作者明确承认的局限

No explicit author-acknowledged limitation was found in the supplied source.

## 13 批判性分析

| `[Analysis]` 观察 | 潜在问题或替代解释 | 为什么重要 | 如何检验 | 依据 |
|---|---|---|---|---|
| 摘要使用“necessary and sufficient” | 可能与当前目标 theorem 大面积重合 | 决定 novelty 生死 | 获取全文并逐一定理比较量词与 kernel 条件 | Publisher abstract |
| 摘要没有 control/policy 语言 | 受控扩展可能非直接 corollary，也可能只是逐 action lifting | 决定是否存在真正新 theorem | 检查定理能否在 action-labelled kernel 上逐 action 应用 | 摘要缺项 + 当前推导 |
| Passage-time characterization 不等于任意 property trace equivalence | 当前 return-law 反例可能躲开 lumpability，却失去自主 Markov 控制状态 | 避免把逻辑严格性误当可部署表示 | 区分 pathwise recursion 与 representative-independent abstract kernel | 当前 kill-gate Proposition 1/2 |

## 14 学到的知识

### Agent-derived knowledge candidates

- First-passage time 不是 recurrence-abstraction 文献中的空白对象；它早已被用来
  刻画 semi-Markov lumpability。
- “路径性质相同”与“聚合过程仍是 Markov/semi-Markov”是两个不同层级。
- 对当前项目，真正必须检查的不是 passage-law 等式本身，而是它是否产生一个
  representative-independent controlled kernel。

## 15 与现有知识的连接

- **Serfozo (1971)：**本文摘要明确说重解释了 Serfozo 的必要充分条件。
- **Bradley (2002)：**外部核验的作者摘要明确提出 passage-time-preserving
  semi-Markov equivalence，说明 property-specific aggregation 已形成后续路线。
- **Probabilistic trace vs bisimulation：**现代概率系统文献区分 trace-level
  preservation 与 branching/bisimulation preservation；当前五状态反例正是该区别的
  return-time 版本。
- **当前项目：**若只保留 return trace，可严格弱于 lumpability；若要求抽象状态具有
  独立控制核，则返回 lumpability 条件。

## 16 研究构想

### Agent-derived research candidates

#### 候选 A：受控逐-action lifting 的精确边界

- 来源：摘要未说明 control/policy quantification。
- 核心假设：某些 policy-class-restricted passage properties 可能弱于逐 action
  controlled lumpability。
- 相对本文的增量：从无控制状态 partition 转向 independently defined policy class。
- 初始方法：先证明是否存在非 nuisance、非 autonomous-kernel 的控制价值增量。
- 如何验证：给出不能由 action-labelled Sumita--Rieders 条件推出的 theorem 或反例。
- 可能失败：逐 action 应用经典定理即可得到全部结果。
- 创新状态：**partially checked; current exact-control-state version failed the kill gate**。

#### 候选 B：结构化尾部族下的有限样本认证

- 来源：经典 exact theorem 不处理从 censored partial trajectories 学习尾性质。
- 核心假设：已知 geometric/hazard/drift envelope 时，可有限样本认证某种
  recurrence-preserving merge。
- 相对本文的增量：统计学习问题，而非 exact transform characterization。
- 初始方法：先固定 tail class、coverage、censoring 和 policy class，再求上下界。
- 如何验证：matching upper/lower sample-complexity bounds。
- 可能失败：结构假设本身已等价于预先知道 positive recurrence，或证书只能覆盖
  训练策略而非控制类。
- 创新状态：**unverified**。

## 来源边界

主来源：
<https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/first-passage-times-and-lumpability-of-semimarkov-processes/F5601F9E5DF2EC1C3259DDE88907E665>

本文卡不声称完成全文精读。若后续获得合法全文，应替换本卡中所有
“不可评估”项，并对 theorem numbering、assumptions、initial-distribution
quantifiers、weak/strong lumpability 和 controlled lifting 重新审计。
