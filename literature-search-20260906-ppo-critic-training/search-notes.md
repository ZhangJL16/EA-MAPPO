# 检索与阅读溯源

时间：2026-09-06。仅公开关键词检索，没有上传项目源码或未发表结果。
用户明确要求深入正文，因此主工作是下载9篇论文和PPG补充材料，查看方法、
证明条件与实验消融；并逐行核对Spinning Up update相关源码。

公开检索词：PPO implementation matters separate value network optimization
early stopping KL critic training；What Matters In On-Policy Reinforcement
Learning；Phasic Policy Gradient auxiliary phase value function；PPO constrained
reinforcement learning FOCOPS CUP P3O；Embedding Safety into RL；Constrained
Policy Optimization；Responsive Safety PID Lagrangian；Projection-Based
Constrained Policy Optimization；A Closer Look at Deep Policy Gradients。

来源：PMLR、NeurIPS官方论文、arXiv、OpenReview以及Spinning Up官方源码。
搜索返回的博客、Reddit、聚合摘要未作为结论依据；应用了来源质量排除规则。
OpenReview部分页面触发浏览器验证，重点内容使用带会议标识的arXiv全文。
C-TRPO的GitHub托管PDF不能被web解析，已通过官方PMLR链接下载并本地解析。
环境没有pdftotext，用已安装pypdf生成带PDF页号的文本；原始PDF保留。

去重后的重点9篇见papers.md，另筛选PID-Lagrangian、P3O、PCPO、ESPO、
A Closer Look at Deep Policy Gradients、Understanding Policy Gradient等候选。
这些候选本轮未全部深读，不据摘要作适配性结论，不给虚构评分。

尚未知：critic补足预算是否改善导航；500场景最终效果；两个续训随机流之外
的稳定性；gamma1有限任务约束的严格政策改进界；神经网络可行域不变性。
上述问题放在实验结果及后续理论研究，不作为启动前的gate链。

交接：实验协议docs/RECOVERY_CRITIC_COMPLETION_PROTOCOL.md；四个分支按
预注册计划运行，首个checkpoint健康后等用户要求评估。文献阅读没有引入
额外启动审批或安全策略模块。
