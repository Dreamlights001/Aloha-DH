# Dual-Arm Sorting Robot MDH Simulation (Python)

该项目用于机械设计前期运动学验证，默认采用 **ALOHA ViperX-300 6DOF 官方参数**，并支持：
- Matplotlib 动画导出
- PyBullet 实时交互拖拽（任意关节点 IK）
- Windows / macOS / WSL / Ubuntu 运行兼容

## 1. 官方参数来源

- ALOHA ViperX-300 6DOF：
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/avx300s.html
- X-Series 连杆尺寸（A/B/C/D/E）：
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications.html
- 参考 VX300S 6DOF：
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html

默认模型：`aloha_vx300s_6dof`

## 2. 环境安装

```bash
conda env create -f environment.yml
conda activate DH
# 或 source activate DH
```

## 3. 跨平台运行命令

脚本已内置 `src` 路径处理，**不需要再手动设置 `PYTHONPATH`**。

### Windows (PowerShell)

```powershell
python .\examples\run_sorting_demo.py --viz matplotlib
python .\examples\run_sorting_demo.py --viz pybullet
python .\examples\run_sorting_demo.py --viz both --save demo.gif
```

### macOS / Ubuntu / WSL

```bash
python examples/run_sorting_demo.py --viz matplotlib
python examples/run_sorting_demo.py --viz pybullet
python examples/run_sorting_demo.py --viz both --save demo.gif
python examples/run_sorting_demo.py --viz pybullet --kinematics poe
```

输出默认落到仓库 `./output/`。例如 `--save demo.gif` -> `./output/demo.gif`。

## 4. 可视化模式说明

- `--viz matplotlib`：导出动画（默认 `./output/sorting_demo.gif`）
- `--viz pybullet`：实时交互
- `--viz both`：先 PyBullet，再导出 Matplotlib 动画
- `--kinematics poe|mdh`：默认 `poe`（官方 ALOHA/VX300S 螺旋轴模型）
- `--kinematics-print-step N`：每 N 帧在终端打印一次运动学矩阵
- `--kinematics-prefix name`：设置 `/output/name_left|right.(csv|json)` 前缀

PyBullet 交互按键：
- `Space` 暂停/继续
- `N` 单步（暂停时）
- `R` 重置帧
- `Tab` 切换当前拖拽控制臂（left/right）
- `Q` / `Esc` 退出
- 鼠标左键：点击并拖拽当前机械臂关节点，实时求解局部 IK

无 GUI 环境（如部分 Ubuntu Server/WSL 无图形）会自动回退 PyBullet `DIRECT` 模式并单次回放，不会卡住。

## 5. 场景元素定义

- 传送带：绿色矩形带面
- 正常件：绿色立方体（静止）
- 瑕疵件：红色正四面体（静止）
- 左机械臂：橙色加粗连杆
- 右机械臂：黄色加粗连杆
- 关节点：灰色
- 关节轴模式（ALOHA 6DOF）：`z/y/y/x/y/x`（waist/shoulder/elbow/forearm_roll/wrist_angle/wrist_rotate）

## 6. 运动学输出

程序默认会把各自由度运动学数据输出到 `/output`：
- `kinematics_left.csv` / `kinematics_right.csv`
- `kinematics_left.json` / `kinematics_right.json`

包含以下机器人学符号字段：
- 相对变换：`^{i-1}T_i`
- 基座到各关节/末端：`^0T_i`
- 基座下关节点坐标：`^0p_i`
- 坐标变换示例：`^ip_ref = ^iT_0 * ^0p_ref`

## 7. WSL / Linux 图形注意事项

若出现 `cannot connect to X server`：
- Windows 11 + WSLg：确保在支持 GUI 的 WSL 会话中运行
- 非 WSLg：安装并启动 X Server（如 VcXsrv），并正确设置 `DISPLAY`
- 纯无头环境：使用 `--viz matplotlib` 或 `--viz pybullet --pybullet-direct`
- 若你在可视化终端中明确需要 GUI，可加 `--force-gui`

## 8. 验证

```bash
# macOS / Ubuntu / WSL
make test
make verify
```

`make verify` 会执行单元测试并导出动画到 `./output/demo.gif`。

Windows 无 `make` 时可直接运行：

```powershell
python -m unittest discover -s tests -v
python .\examples\run_sorting_demo.py --viz matplotlib --products 8 --save demo.gif
```

## 9. 关键文件

- `src/dual_arm_sim/config.py`：官方模型参数与单位转换
- `src/dual_arm_sim/platform_utils.py`：平台/GUI 探测与 Matplotlib 运行时适配
- `src/dual_arm_sim/scenario.py`：Matplotlib 与 PyBullet 双模式可视化
- `examples/run_sorting_demo.py`：跨平台统一入口
- `tests/test_dual_arm_sim.py`：测试集合
