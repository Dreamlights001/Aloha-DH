# Dual-Arm Sorting Robot MDH Simulation (Python)

用于毕业设计前期机械方案验证的双臂最简运动学仿真，已切换为 **ALOHA ViperX-300 6DOF** 官方参数（长度与关节范围）。

## 1. 官方参数来源

- ALOHA ViperX-300 6DOF 规格：
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/avx300s.html
- X-Series 连杆尺寸（A/B/C/D/E）：
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications.html
- 参考 6DOF 规格页：
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html

默认模型：`aloha_vx300s_6dof`

- 关节范围（deg）：
  `[-180,180], [-108,114], [-123,92], [-180,180], [-100,123], [-180,180]`
- 关键段长（mm）：
  `upper_arm=306, forearm=300, wrist=70, gripper_to_rail=69, finger_tip=68`

## 2. 目录

- `src/dual_arm_sim/config.py`：官方模型参数、单位转换、MDH 构建
- `src/dual_arm_sim/robot.py`：FK / IK / 双臂轨迹规划
- `src/dual_arm_sim/scenario.py`：Matplotlib 动画 + PyBullet 实时交互
- `examples/run_sorting_demo.py`：统一入口（`--viz` / 输出路径 / 报告）
- `tests/test_dual_arm_sim.py`：单测

## 3. 环境

```bash
conda env create -f environment.yml
conda activate DH
# 或 source activate DH
```

## 4. 运行与输出（WSL/Win11）

默认动画输出目录为仓库内 `./output`。

```bash
# 导出动画（默认保存到 ./output/sorting_demo.gif）
PYTHONPATH=src python examples/run_sorting_demo.py --viz matplotlib

# 实时交互（PyBullet）
PYTHONPATH=src python examples/run_sorting_demo.py --viz pybullet

# 先交互再导出
PYTHONPATH=src python examples/run_sorting_demo.py --viz both --save demo.gif
```

说明：`--save demo.gif` 会自动解析为 `./output/demo.gif`。

## 5. PyBullet 交互控制

- `Space`：暂停/继续
- `N`：单步（暂停时）
- `R`：重置帧
- `Q` 或 `Esc`：退出

WSL 下若出现 `cannot connect to X server`：
- Windows 11 + WSLg：确保通过支持 GUI 的终端启动 WSL
- 非 WSLg：安装并启动 X Server（如 VcXsrv），并正确配置 `DISPLAY`

## 6. 验证

```bash
make test
make verify
```

`make verify` 会执行单测，并导出 `./output/demo.gif`。

## 7. 参数调优建议

主要修改 `src/dual_arm_sim/config.py`：
- `ARM_OFFSET`：双臂基座间距
- `CONVEYOR_CONFIG` / `SORTING_BINS`：产线布局
- `build_mdh_params_from_official_specs()`：如需改为你的机械草案参数映射

每次调参建议看终端 `waypoint error`，优先保持 `failed=0` 且最大误差低于 `0.03m`。
