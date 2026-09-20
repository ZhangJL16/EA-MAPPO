# 旧研究资产统一归档

所有从 /home/zjl 父目录收回的旧研究资产，均放在 `parent_workspace_20260920/`。
此目录只供查阅和复现，不是新研究入口，不自动运行其中任何历史任务。

保留内容：article 独立项目、控制适应 checkpoint、两个 Python 环境、DAD 和
safe-control-gym 参考源码、FPL/旧实验压缩包、结项迁移包和 startup 报告。
每项原路径/新路径见 `../docs/cleanup_20260920/parent_relocation.json`。
本次同文件系统 rename，迁移前后 inode/元数据清单一致；未删除数据或复制大型依赖。

## 复现说明

- 两个归档 Python 环境内部可能仍引用旧绝对路径，不能保证移动后直接运行。
- 若要原样复现，把所需目录按 relocation 清单恢复到原位置；恢复时先确认目标不存在。
- 也可以按项目 requirements 与 Python 版本重建环境，再修正实验配置中的路径。
- `.elan`、`.local` 等全局工具仍在 /home/zjl，不在本归档内；归档并非完全自包含运行环境。
- `research_archive_20260920/remaining_research_records.tar.gz` 含私有 seed-custody 信息，
  权限已保留，不要作为公开材料直接上传。
- `mappo_closeout_20260920.tar.gz` 是 c1da2fe3 的旧结项快照，不包含后来的清理记录。
- 大型归档目录不进入 Git；需要搬整套资产时，务必连同本机 archive 目录复制。
