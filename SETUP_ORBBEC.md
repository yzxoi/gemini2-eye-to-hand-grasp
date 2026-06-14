# Orbbec Gemini 2 改造说明

原课程使用 Intel RealSense D435（`pyrealsense2`）。本项目已改为使用 **Orbbec Gemini 2**
深度相机（`pyorbbecsdk2`）。本文件记录改动内容与运行方法。

## 1. 改动概览

| 文件 | 说明 |
|------|------|
| `orbbec_utils.py` | **新增**。封装 `pyorbbecsdk`，提供 `OrbbecCamera` 类与 `frame_to_bgr_image()`。负责开流、深度对齐到彩色（D2C）、读取内参、反投影。同时兼容 pyorbbecsdk v1/v2 的高层 API。 |
| `4.3d_calibrate_test.py` | 去掉 `pyrealsense2`，改用 `OrbbecCamera`。深度无效（=0）时跳过该帧，避免污染标定。 |
| `1.test_d435_aruco.py` | 同样移植到 Orbbec，作为相机/检测/覆盖范围的验证工具。 |
| `5.dino_detect.py` | 去掉 `pyrealsense2`，改用 `OrbbecCamera`。Grounding DINO 零样本检测可视化。 |
| `6.dino_grasp.py` | 去掉 `pyrealsense2`，改用 `OrbbecCamera`。另外：DINO 模型改为**只加载一次**（原版逐帧重载 ~700MB 模型，CPU 上无法用）；检测前 BGR→RGB；修正 `running`→`self.running`。 |

> **transformers 5.x API 变更**：`post_process_grounded_object_detection` 的
> `box_threshold` 参数已改名为 `threshold`；结果里的字符串标签从 `labels`
> 移到了 `text_labels`（`labels` 现在是数字索引）。脚本 5/6 已按新 API 更新。

> 标定数学不受影响：`T_camera2base = base_coords @ pinv(cam_coords)` 为线性最小二乘，
> 会自动吸收单位缩放；标定与抓取都走同一个 `get_aruco_center()`，尺度一致即可。

## 2. 运行环境（已在本机搭好）

- 相机：Orbbec Gemini 2（USB `2bc5:0670`，已识别，udev 规则已就绪）
- 虚拟环境：`.venv/`（Python 3.12，由 `uv` 创建）
- 关键依赖：
  - `pyorbbecsdk2==2.1.1`（PyPI 预编译 wheel）
  - `opencv-contrib-python==4.10.0.84`（注意：必须是 **contrib** 版且为 4.10，
    新版 4.13 移除了 `aruco.estimatePoseSingleMarkers`，且普通版无 aruco）
  - `numpy>=2.1`

### 如需在新机器重建环境

```bash
cd /home/yzxoi/Downloads/3d_grasp
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pyorbbecsdk2
# 修正 opencv：pyorbbecsdk2 会拉入普通版 opencv-python，需替换为 contrib 4.10
uv pip uninstall --python .venv/bin/python opencv-python
uv pip install --python .venv/bin/python "opencv-contrib-python==4.10.0.84"
```

若相机无权限（普通用户打不开），安装 udev 规则（需 sudo）：
```bash
# pyorbbecsdk 仓库 scripts/install_udev_rules.sh，或 OrbbecSDK 自带脚本
sudo bash install_udev_rules.sh && sudo udevadm control --reload && sudo udevadm trigger
```

## 3. 运行命令

```bash
cd /home/yzxoi/Downloads/3d_grasp

# 相机/检测验证（无需机械臂）
.venv/bin/python 1.test_d435_aruco.py

# 手眼标定（需机械臂在线，默认 localhost:12345）
.venv/bin/python 4.3d_calibrate_test.py --visualize --calibrate

# 抓取测试
.venv/bin/python 4.3d_calibrate_test.py --visualize --recognize

# Grounding DINO 零样本检测（无需机械臂，控制台输入类别，q 退出）
.venv/bin/python 5.dino_detect.py

# Grounding DINO 检测 + 自动抓取（需机械臂在线 + 已完成标定 camera2base.npy）
.venv/bin/python 6.dino_grasp.py
```

### DINO 相关依赖（已装在 `.venv`）

```bash
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/bin/python transformers
```

> 本机无 NVIDIA GPU，装的是 **CPU 版 torch**，DINO 推理在 CPU 上较慢（每帧数秒级）。
> 模型权重在本地 `./grounding-dino-base/`，无需联网下载。

标定结果保存在 `save_parms/camera2base.npy`。

## 4. 已知提示

- 控制台可能出现 `Timestamp anomaly detected`：Gemini 2 固件的**非致命**日志，不影响取流。
  `orbbec_utils.py` 已将 SDK 控制台日志级别调到 FATAL 以减少刷屏。
- 深度单位：`OrbbecCamera.wait_frames()` 返回的 `depth_m` 为**米**（与原 RealSense
  `get_distance()` 一致）。
- 实测内参（1280×720）：fx≈690.5, fy≈690.7, cx≈639.1, cy≈352.0。
- **是否需要相机内参标定（脚本 2/3）？不需要。** 与文档 2.1.6 的结论一致：
  应直接使用 SDK 出厂内参做反投影，自己用棋盘格标定出的 K 反而与 SDK 输出图不匹配。
  脚本 2/3 仅作教学/排查用途（且仍是 RealSense 版，未移植）。
- **畸变取 0**：Gemini 2 上报 8 参数 rational 畸变模型 (k1..k6,p1,p2)，实测
  k1≈k4、k2≈k5、k3≈k6 → 分子≈分母 → 净畸变 ≈ 0（边角约 0.5%），彩色图已基本矫正。
  因此 `orbbec_utils.py` 统一用 `dist=0` + 针孔反投影（与课程一致）。
  ⚠️ 切勿只截取前 5 个 rational 系数 `[k1,k2,p1,p2,k3]` 当作畸变传入 OpenCV——
  缺了分母会造成严重的假畸变。

## 5. 6d 棋盘格手眼标定（眼在手外）

`6d/` 原本是**眼在手上**（相机装在末端）的棋盘格手眼标定，输出 `T_camera2end`。
本机是**眼在手外**（相机固定、棋盘格装在末端），已改造为输出刚体
`T_camera2base`（相机→基座，单位 mm）。

改动：
- `1/2/3/4` 脚本相机由 RealSense 改为 `OrbbecCamera`（脚本会把仓库根目录加入
  `sys.path` 再 `from orbbec_utils import OrbbecCamera`）。`0.teach_mode.py` 不涉及相机。
- 眼在手外转换在 `3.calibrate.py`：把机器人位姿取逆（base→end）喂给
  `cv2.calibrateHandEye`，返回的即 cam→base。输出 `6d/T_camera2base.yaml`。
- `4.test_gripper.py` 直接用固定的 `T_camera2base` 做 `P_base = T_camera2base @ P_camera`，
  不再需要 `get_T()`。
- `6d/episodeApp.py` 的 `roboticstoolbox`/`spatialmath` 改为可选导入（标定流程用不到）。

物理摆放：相机固定三脚架对准工作区；棋盘格刚性固定在机械臂末端，随臂运动。

运行（**在 `6d` 目录下**，注意用上一级的 venv）：
```bash
cd 6d
../.venv/bin/python 1.generate_points.py prepare    # 机械臂到初始位
../.venv/bin/python 1.generate_points.py generate   # 自由模式手托，空格存点，s 保存退出
../.venv/bin/python 2.generate_images_and_T.py      # 自动跑点，存棋盘格图 + T_end2base
../.venv/bin/python 3.calibrate.py                  # 选择算法(1-3)，输出 T_camera2base.yaml
../.venv/bin/python 4.test_gripper.py               # 用 ArUco 验证标定精度
```

> graspnet-baseline 的眼在手外改造与环境（CUDA 扩展编译等）属于下一阶段，尚未开始。
