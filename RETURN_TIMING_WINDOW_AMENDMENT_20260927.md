# 固定滞后返站分支的同任务边界勘误

原 `RETURN_TIMING_WINDOW_PROTOCOL_20260927.md` 和 211-root manifest 保留不改。
执行中发现选择器检查了 `completed_targets[start:takeover]`，但没有检查 root **之前**的完成计数。因此，若任务恰好在 root 决策完成，后续计数恒定也会误收一个跨任务窗口。

在任何滞后矩阵汇总前，对全部 211 个冻结 root 的归档 `events.jsonl` 执行相同的边界审计：要求 root 决策前的 `completed_targets` 与首次接管决策后的计数相等。只有 `window_independent_seeds_s102_m038_lag05` 不满足（0 → 1），且该 root 从未生成有效分支结果。将它标为 protocol-ineligible，不补抽。有效固定滞后 root 数为 210；各滞后计数为 48、46、44、42、30。

这是一处取样实现勘误，不根据返站时间、能耗、奖励或安全结果筛选。所有既有有效分支保持原冻结 manifest 的 SHA256 关联；排除名单、事件哈希及计数独立归档。执行器仍对每个有效 root 做精确历史重放和两分支克隆。汇总必须同时报告 211 个原冻结 root、1 个勘误排除和 210 个有效结果。不能把相邻滞后状态当成独立 episode。
