# 独立分支与 ARM64 三维实验交接

目标远程仓库：`ZhangJL16/EA-MAPPO` 的独立分支 `research/uav-learning-20260926`。该分支是 `/home/zjl/uav_learning_research` 独立项目的源码快照，不继续旧 `master` 的 regenerative-control 实验。用户已说明：本地提交代码到 GitHub，服务器侧有人负责 ARM64 可运行性适配；用户之后在 Ascend 服务器运行三维研究。服务器资源快照见 `ASCEND_SERVER_INVENTORY_20260926.md`。

本次交付包括 `nav3d/`、`nav3d_v2/`、`delivery_1km/`、`dual_constraint_2d/`、`gpu2d/`、测试、`pyproject.toml`、`uv.lock` 和研究协议。干净检出测试发现，冻结的三维 v2 资格脚本与 1 km 能耗审计需要原始三维 manifest/航迹；因此分支**选择性包含** `artifacts/nav3d_qualification_20260923/`、`nav3d_v2_paired_20260923/`、`nav3d_v2_holdout_20260923/`、`nav3d_v2_freeze_20260923.json`、`delivery_1km_nav_v1_20260923/`、`delivery_1km_energy_oat_20260925/` 和 `delivery_1km_preflight_cv_20260925/`，共约 17 MiB。其余大规模二维/GPU 运行产物、`.venv/`、性能剖析缓存和 `literature/` 下的论文全文与提取文本不进入 GitHub。

三维当前可执行范围：解析导航 v2 与 1 km 静态地图导航资格。已在本地完成的 1 km v1 资格为 96/96 到达、0 统一接触；这不是完整三维配送或能量安全结论。当前项目尚无持续订单队列、取货→送达履约状态机、真实 M100 功率曲线或三维高层学习训练入口。`sac_native_qualification.py` 仍依赖旧 `/home/zjl/mappo` 和冻结 SAC 资产，本分支未携带那些资产，故不作为服务器三维解析实验入口。

服务器侧按实际仓库权限克隆该分支、适配 ARM64 依赖后，在项目根目录执行下列**单行命令**。现有三维导航只需要 NumPy、SciPy、OSQP；`uv.lock` 应以 ARM64 环境复核，不应通过修改已冻结的导航源码来掩盖兼容问题。

```bash
uv sync --frozen
```

```bash
uv run python -m unittest discover -s tests -p 'test_nav3d*.py'
```

```bash
uv run python -m unittest discover -s tests -p 'test_delivery_1km_*.py'
```

只有在后续明确启动三维资格运行时，才使用下面的已有可恢复入口；`--stop-after-checkpoint` 只完成首个检查点（2 个航段），检查健康后再决定是否续跑。输出目录必须是新的空路径。

```bash
uv run python -m delivery_1km.qualify --output artifacts/delivery_1km_nav_arm_v1 --stop-after-checkpoint
```

```bash
uv run python -m delivery_1km.qualify --output artifacts/delivery_1km_nav_arm_v1 --resume
```

`delivery_1km.qualify` 当前按航段顺序执行，不会自动占满服务器 40 个 CPU。若后续要做大规模三维矩阵，应先在独立版本中设计多地图并行、内存上限、断点与配对结果协议；本次只交付当前可运行代码，不启动服务器实验。CUDA 专用的 `gpu2d` 不因 ARM64 适配而自动支持 Ascend NPU。
