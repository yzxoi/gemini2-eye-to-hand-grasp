# 6D 手眼标定 + GraspNet 抓取（眼在手外 / Gemini 2）说明

本文件记录 `6d/`（棋盘格手眼标定）与 `graspnet-baseline/`（6-DoF 点云抓取）
两部分从 RealSense + 眼在手上 改造为 **Orbbec Gemini 2 + 眼在手外（eye-to-hand）**
的内容、环境搭建与运行方法。配合相机基础说明见 `SETUP_ORBBEC.md`。

整体流程：
```
6d 棋盘格手眼标定  ──>  6d/T_camera2base.yaml（相机->基座 刚体, mm）
                                   │
                                   ▼
graspnet 抓取（1.verify_grasp / VLM 2+3）用该矩阵把相机系抓取位姿转到基座系
```

> **眼在手外**：相机固定在外部三脚架（不在机械臂上）。`T_camera2base` 是**固定常量**，
> 不随机械臂运动变化。原课程是眼在手上（相机在末端，`T_camera2end` + 实时 `T_end2base`），
> 已全部改为读取固定的 `T_camera2base`。

---

## 1. 环境（已在本机 `.venv` 内搭好）

在 `SETUP_ORBBEC.md` 基础上，本阶段新增：

- **GPU 加速**：NVIDIA RTX 4060 + 驱动 580 + `torch 2.12.0+cu130`（CUDA 13）。
- **CUDA 13 编译器**：从 NVIDIA apt 源装了最小组件 `cuda-nvcc-13-0`、`cuda-cudart-dev-13-0`
  （在 `/usr/local/cuda-13.0`）。完整 math 头文件（cusparse/cublas/cusolver）直接复用
  torch 自带的 `…/site-packages/nvidia/cu13/include`。
- **Python 依赖**：`open3d`、`graspnetAPI`(本地源码装)、`scipy`、`scikit-learn`、
  `trimesh`、`transforms3d`、`cvxopt`、`matplotlib`、`autolab_core`、`openai`、`ollama`。
- **opencv 仍须是 `opencv-contrib-python==4.10.0.84`**：`autolab_core` 会拉入普通版
  `opencv-python==4.13`（无 aruco、无 estimatePoseSingleMarkers），装完务必复位：
  ```bash
  uv pip uninstall --python .venv/bin/python opencv-python opencv-contrib-python
  uv pip install  --python .venv/bin/python "opencv-contrib-python==4.10.0.84"
  ```

### 1.1 CUDA 扩展（pointnet2 / knn）已重新编译

原仓库自带的 `.so` 是 **Python 3.6** 的，无法在本机 py3.12 加载，已针对
**CUDA 13 / torch 2.12 / sm_89** 重新编译并打了源码补丁：

- pointnet2：`.data<T>()` → `.data_ptr<T>()`，`x.type().is_cuda()` → `x.is_cuda()`。
- knn：移除已废弃的 `THC/THC.h` 与 `THCState`，`THCudaMalloc/Free` 改 `cudaMalloc/Free`，
  `.data<T>()` → `.data_ptr<T>()`，gencode `sm_86` → `sm_89`（4060 是 Ada）。

如需重新编译（换机/换 torch 时）：
```bash
cd /home/yzxoi/Downloads/3d_grasp/graspnet-baseline
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export CPATH=/home/yzxoi/Downloads/3d_grasp/.venv/lib/python3.12/site-packages/nvidia/cu13/include
cd pointnet2 && rm -rf build dist *.egg-info && ../../.venv/bin/python setup.py install
cd ../knn   && rm -rf build dist *.egg-info && ../../.venv/bin/python setup.py install
```
自检（应输出 `ALL-GOOD`）：
```bash
cd /home/yzxoi/Downloads/3d_grasp/graspnet-baseline
../.venv/bin/python -c "
import sys,os; [sys.path.append(p) for p in ['models','dataset','utils','pointnet2','knn']]
import torch, pointnet2._ext, knn_pytorch.knn_pytorch
print('ext OK, cuda', torch.cuda.is_available())"
```

---

## 2. 第一步：眼在手外手眼标定（`6d/`）

**物理摆放**：相机固定三脚架对准工作区；**棋盘格刚性固定在机械臂末端**，随臂运动。
棋盘格参数在 `6d/config.yaml`（`pattern_size`、`square_size` 必须与实物一致）。

在 **`6d` 目录下**运行（注意用上一级的 venv，需机械臂在线 `localhost:12345`）：
```bash
cd /home/yzxoi/Downloads/3d_grasp/6d

# ① 移动到初始位
../.venv/bin/python 1.generate_points.py prepare
# ② 自由模式手托机械臂，让固定相机看到末端棋盘格；空格存点（需检测到角点），s 保存退出
../.venv/bin/python 1.generate_points.py generate
# ③ 按存好的角度自动跑一遍，保存棋盘格图 + 每个位姿的 T_end2base
../.venv/bin/python 2.generate_images_and_T.py
# ④ 计算标定，选择算法(1 Horaud / 2 Tsai / 3 Park)，输出 6d/T_camera2base.yaml
../.venv/bin/python 3.calibrate.py
# ⑤ 用 ArUco 验证：把标记放工作区，机械臂应能准确吸取（验证 T_camera2base 精度）
../.venv/bin/python 4.test_gripper.py
```

- 采集建议 15~20 个姿态，棋盘格在画面里位置/角度尽量分散，重投影误差越小越好。
- `3.calibrate.py` 关键改动（眼在手外）：把机器人位姿取逆（base→end）喂给
  `cv2.calibrateHandEye`，返回的即 **相机→基座**，保存到 `6d/T_camera2base.yaml`（mm）。
- `4.test_gripper.py` 用固定 `T_camera2base` 做 `P_base = T_camera2base @ P_camera`。

> 该 `T_camera2base.yaml` 是后续 GraspNet 抓取的唯一标定依赖，务必先标好、用 ④ 验证通过。

---

## 3. 第二步：GraspNet 6-DoF 抓取（`graspnet-baseline/`）

所有脚本在 **`graspnet-baseline` 目录下**运行，需：机械臂在线、Gemini 2 已接、
GPU 可用、`checkpoint-rs.tar` 在当前目录、`6d/T_camera2base.yaml` 已生成。

### 3.1 点云直接抓取（无需 VLM）：`1.verify_grasp.py`

整桌点云送入 GraspNet，输出候选抓取，open3d 窗口可视化（左：全部，右：竖直筛选后），
按键交互后移动机械臂抓取。
```bash
cd /home/yzxoi/Downloads/3d_grasp/graspnet-baseline
../.venv/bin/python 1.verify_grasp.py
```
open3d 窗口按键：`W/S` 抬升高度、`A/D` 角度、`M` 依次尝试抓取（带逆解判断）、
`J/K` 夹爪开合、`I/U` 角度阈值、`O/P` 高度阈值、`Q` 退出。

### 3.2 自然语言抓取（VLM 两进程）：`2.demo_VLM_grasp.py` + `3.demo_VLM_handler.py`

- `2.*`（前端）：持有相机、显示画面、接收你输入的自然语言（如“把香蕉放进盒子里”），
  调用云端 VLM/DINO 得到目标框，把彩色图/深度图/内参/目标框存到 `VLM_related/`。
- `3.*`（后端）：监听 `VLM_related/exchange/result_box.npy`，对框内区域跑 GraspNet 并抓取。

需先设置阿里云 DashScope Key（`2.*` 用到云端大模型）：
```bash
export DASHSCOPE_API_KEY=你的key
```
开两个终端（都在 `graspnet-baseline` 下）：
```bash
# 终端 A（后端，先起，加载模型并等待）
../.venv/bin/python 3.demo_VLM_handler.py
# 终端 B（前端，输入自然语言指令）
../.venv/bin/python 2.demo_VLM_grasp.py
```

眼在手外改动（2/3 与 1 一致）：加载固定 `../6d/T_camera2base.yaml`，
`T_grasp2base = T_camera2base @ T_grasp2camera`，再扣除舵机夹爪偏置 `T_servo2end`。
另外把过时的 transformers API 修正为 `threshold=` 和 `text_labels`。

---

## 4. 关键改动与坑（务必了解）

- **标定依赖路径**：graspnet 脚本按 `仓库根/6d/T_camera2base.yaml` 读取标定结果，
  跑抓取前必须先完成第 2 节标定。
- **深度单位**：相机给的 `depth_m`（米）统一 `*1000` 转毫米喂给 GraspNet
  （`factor_depth=1000` → 还原成米），与原 RealSense z16(mm) 口径一致。
- **`torch.load` weights_only**：torch≥2.6 默认 `weights_only=True` 会让旧 checkpoint
  加载失败，脚本已显式 `weights_only=False`。
- **`workspace_mask`**：`1.verify_grasp.py` 用 `doc/example_data/workspace_mask.png`
  限定桌面工作区，是按原相机视角做的。换了相机摆位后，**若抓取点过滤异常需重做该 mask**
  （白=工作区，分辨率 1280×720）。
- **`filter_vertical_grasps` 高度过滤**用 `t_camera2base[2]`（相机在基座系的高度），
  眼在手外下它直接来自 `T_camera2base`，标定不准会影响过滤，请先用 `4.test_gripper.py` 验收。
- **opencv 必须 4.10 contrib**（见 1 节）；**机械臂须在线**（`localhost:12345`）；
  **相机别被别的进程占用**。
- VLM 脚本需要 `DASHSCOPE_API_KEY`（付费云端）；不设置 `2.*` 会直接报错退出。
