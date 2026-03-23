# Dual-Arm Sorting Robot MDH Simulation (Python)

该项目用于机械设计前期运动学验证，默认采用 **ALOHA ViperX-300 6DOF 官方参数**，并支持：
- Matplotlib 动画导出
- PyBullet 实时交互拖拽（任意关节点 IK）
- Xbox / 盖世小鸡等手柄交互（pygame）
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
- `--input-device auto|mouse|gamepad`：交互输入策略
- `--gamepad-enable / --no-gamepad`：启用/禁用手柄
- `--ui-style industrial|minimal`：交互 HUD 风格
- `--drag-mode default_drag|default_normal`：默认鼠标模式
- 右侧 `Params` 栏：
  - `Mode Toggle [click]`：点击切换 `DRAG/NORMAL`
  - `Mode State`：当前模式状态（0=Normal，1=Drag）
  - `L-J1..J6 / R-J1..J6` 的 `^0p_i` 坐标（x/y/z，米，实时刷新）

PyBullet 交互按键：
- `Space` 暂停/继续
- `N` 单步（暂停时）
- `R` 重置帧
- `Q` / `Esc` 退出
- 鼠标左键：点击并拖拽任意机械臂关节点，实时求解局部 IK
- 鼠标右键拖拽：旋转视角
- 鼠标滚轮：缩放
- `W/A/S/D`：俯仰/偏航微调；`Z/X`：距离微调
- `M`：切换鼠标普通模式 / 拖拽模式（与 Params 栏模式开关联动）

手柄默认映射（Xbox / GameSir 兼容）：
- 左摇杆：平移视角目标（拖拽时用于目标微调）
- 右摇杆：旋转视角
- 扳机：缩放
- `LB/RB`：左右臂切换
- 方向键：关节编号切换
- `A/B`：开始/取消拖拽
- `X`：切换普通/拖拽模式
- `Start`：暂停/继续

无 GUI 环境（如部分 Ubuntu Server/WSL 无图形）会自动回退 PyBullet `DIRECT` 模式并单次回放，不会卡住。

## 5. 场景元素定义

- 传送带：绿色矩形带面
- 正常件：绿色实心立方体（静止，尺寸缩小到原来的 80%）
- 瑕疵件：红色实心正四面体（静止，尺寸缩小到原来的 80%）
- 左机械臂：橙色加粗连杆
- 右机械臂：黄色加粗连杆
- 关节点：灰色
- 关节轴模式（ALOHA 6DOF）：`z/y/y/x/y/x`（waist/shoulder/elbow/forearm_roll/wrist_angle/wrist_rotate）
- 坐标轴刻度：每 0.25m 一个刻度
- 右侧实时面板：显示左右臂 `J1..J6` 的 `^0p_i=(x,y,z)`
- 工业风 HUD：显示输入源、模式状态、手柄连接状态、按钮化模式切换

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
