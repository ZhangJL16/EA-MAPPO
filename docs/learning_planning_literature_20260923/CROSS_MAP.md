# 跨地图高层调度：文献核查 2026-09-23

用户接受：新地图交给低层导航，高层不接收完整地图，冻结参数零样本部署。仅研究范围决定，不授权实现、改现有运行或训练。固定电池容量，部分充电、动态订单、生成时计时、按时完成数为当前目标；未取货过期退出，取货后逾期仍交付。

UnderMind 搜索并下载三个完整PDF；核读选定正文，不声称穷尽检索或全部证明核验。已导入 Zotero 跨地图迁移与神经路由（36YYY58F），MD5验证通过。

- Kwon et al., Matrix Encoding Networks for Neural Combinatorial Optimization, NeurIPS 2021. PDF pp.2–3,6: 矩阵边特征、双向二部图注意力、ATSP自回归动作选择、POMO RL。随机实例训练，不证明单物理地图训练足够。 https://papers.nips.cc/paper/2021/hash/29539ed932d32f1c56324cded92c07c2-Abstract.html
- Huang et al., Rethinking Light Decoder-based Solvers for Vehicle Routing Problems, ICLR 2025. PDF pp.5–6: 静态嵌入与当前子问题上下文不匹配、轻解码器能力分析及ReLD。主要规模/约束泛化，不是UAV续航定理。 https://proceedings.iclr.cc/paper_files/paper/2025/hash/447ac93bf22099aa346a45577376492d-Abstract-Conference.html
- Son et al., RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization, ICLR 2026. UnderMind仍为2025旧题名，PDF为2503.16159v2 2026-03-14；官网确认正式发表。PDF pp.3–4,6–8,10（文本方法/测试协议/随机VRP段落）：OSRM城市距离与时间矩阵、ANE/NAB、100城数据，80城训练，未见城市OOD测试；还包含随机时变交通测试，不应称其完全没有动态因素。但这些证据不等于本文所需的单地图训练、持续随机新订单、可中断配送、部分充电和载荷能耗。 https://openreview.net/forum?id=sKvo9ZZfpe

推论：从地图生成候选节点间代价矩阵、以RL选择下一节点有直接先例；跨地图泛化本身不是创新。建议表述为面向持续配送的跨地图策略泛化，需实证寻找现有策略在资源与截止时间耦合下的不足。先区分单地图多订单实例->新地图，与多地图->新地图，不能用后者证明前者。

拟议接口：节点包括取货点、送货点、站点和当前位置；节点状态含时限/载货/当前任务；边含时间及依赖载荷、速度状态的能耗估计。边代价不是静态精确真值；需保留路径未找到或估计无效标识。高层直接输出动作，不改成高层MPC。最短路查询不自动求解服务次序，但输入更充分也使传统规划基线更强。
