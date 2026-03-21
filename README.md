# Dual-Arm Sorting Robot MDH Simulation (Python)

该项目用于机械设计前期运动学验证，默认采用 **ALOHA ViperX-300 6DOF 官方参数**，并支持：
- Matplotlib 动画导出
- PyBullet 实时交互
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
```

输出默认落到仓库 `./output/`。例如 `--save demo.gif` -> `./output/demo.gif`。

## 4. 可视化模式说明

- `--viz matplotlib`：导出动画（默认 `./output/sorting_demo.gif`）
- `--viz pybullet`：实时交互
- `--viz both`：先 PyBullet，再导出 Matplotlib 动画

PyBullet 交互按键：
- `Space` 暂停/继续
- `N` 单步（暂停时）
- `R` 重置帧
- `Q` / `Esc` 退出

无 GUI 环境（如部分 Ubuntu Server/WSL 无图形）会自动回退 PyBullet `DIRECT` 模式并单次回放，不会卡住。

## 5. WSL / Linux 图形注意事项

若出现 `cannot connect to X server`：
- Windows 11 + WSLg：确保在支持 GUI 的 WSL 会话中运行
- 非 WSLg：安装并启动 X Server（如 VcXsrv），并正确设置 `DISPLAY`
- 纯无头环境：使用 `--viz matplotlib` 或 `--viz pybullet --pybullet-direct`
- 若你在可视化终端中明确需要 GUI，可加 `--force-gui`

## 6. 验证

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

## 7. 关键文件

- `src/dual_arm_sim/config.py`：官方模型参数与单位转换
- `src/dual_arm_sim/platform_utils.py`：平台/GUI 探测与 Matplotlib 运行时适配
- `src/dual_arm_sim/scenario.py`：Matplotlib 与 PyBullet 双模式可视化
- `examples/run_sorting_demo.py`：跨平台统一入口
- `tests/test_dual_arm_sim.py`：测试集合
