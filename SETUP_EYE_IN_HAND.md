# 眼在手上（Eye-in-hand）标定 + 抓取说明（Gemini 2）

本文件记录 **眼在手上（相机装在机械臂末端，随臂运动）** 的手眼标定与抓取流程，
作为眼在手外（`6d/` + `graspnet-baseline/1.verify_grasp.py`，见 `SETUP_GRASPNET.md`）的对照实现。
相机基础说明见 `SETUP_ORBBEC.md`。

整体流程：
```
6d_eye_in_hand 棋盘格手眼标定  ──>  6d_eye_in_hand/T_camera2end.yaml（相机->末端 刚体, mm）
                                            │
                                            ▼
graspnet 眼在手上抓取（1.verify_grasp_eye_in_hand.py）
  采集瞬间记录 T_end2base，T_grasp2base = T_end2base @ T_camera2end @ T_grasp2camera
```

---

## 眼在手上 vs 眼在手外（关键差异）

| | 眼在手外（`6d/`） | 眼在手上（`6d_eye_in_hand/`） |
|---|---|---|
| 相机位置 | 固定在三脚架（基座系不动） | 装在机械臂末端，随臂运动 |
| 棋盘格位置 | 固定在**末端**，随臂运动 | 固定在**工作台**（基座系不动） |
| 标定输出 | `T_camera2base`（相机->基座，固定） | `T_camera2end`（相机->末端，固定） |
| `calibrateHandEye` 输入 | **末端->基座取逆**（base->end） | **末端->基座直接喂入**（原生用法） |
| 抓取坐标变换 | `P_base = T_camera2base @ P_camera` | `P_base = T_end2base(采集瞬间) @ T_camera2end @ P_camera` |
| 抓取时机械臂 | 移开以免遮挡固定相机 | 移到观察位俯视工作区（该位姿的 T_end2base 参与变换） |

> OpenCV 的 `cv2.calibrateHandEye` 原生求解的就是眼在手上（cam2gripper），
> 所以眼在手上**无需取逆**；眼在手外才需要把位姿取逆。本仓库两种都已验证（合成数据可
> 恢复已知变换到 ~1e-13）。

---

## 1. 环境

与 `6d/` + `graspnet-baseline/` 完全相同（同一个 `.venv`、同样的 CUDA 扩展），
无需额外安装。见 `SETUP_ORBBEC.md` 与 `SETUP_GRASPNET.md`。

`6d_eye_in_hand/` 中 `0.teach_mode.py`、`1.generate_points.py`、`2.generate_images_and_T.py`、
`config.yaml`、`episodeApp.py` 与 `6d/` 内容一致（采集流程通用），只有
`3.calibrate.py`、`4.test_gripper.py` 是眼在手上专属实现。

---

## 2. 第一步：眼在手上手眼标定（`6d_eye_in_hand/`）

**物理摆放**：相机刚性固定在机械臂末端；**棋盘格平放/固定在工作台**（基座系内不动）。
棋盘格参数在 `6d_eye_in_hand/config.yaml`（`pattern_size`、`square_size` 必须与实物一致）。

在 **`6d_eye_in_hand` 目录下**运行（用上一级的 venv，需机械臂在线 `localhost:12345`）：
```bash
cd /home/yzxoi/Downloads/3d_grasp/6d_eye_in_hand

# ① 移动到初始位
../.venv/bin/python 1.generate_points.py prepare
# ② 自由模式手托机械臂，让末端相机从不同角度看到台面上的棋盘格；空格存点（需检测到角点），s 保存退出
../.venv/bin/python 1.generate_points.py generate
# ③ 按存好的角度自动跑一遍，保存棋盘格图 + 每个位姿的 T_end2base
../.venv/bin/python 2.generate_images_and_T.py
# ④ 计算标定，选择算法(1 Horaud / 2 Tsai / 3 Park)，输出 6d_eye_in_hand/T_camera2end.yaml
../.venv/bin/python 3.calibrate.py
# ⑤ 用 ArUco 验证：把标记放工作区，机械臂应能准确吸取（验证 T_camera2end 精度）
../.venv/bin/python 4.test_gripper.py
```

- 采集建议 15~20 个姿态，棋盘格在画面里位置/角度尽量分散，重投影误差越小越好。
- `3.calibrate.py` 眼在手上关键点：把「末端->基座」位姿**直接**（不取逆）喂给
  `cv2.calibrateHandEye`，返回的即 **相机->末端**，保存到 `6d_eye_in_hand/T_camera2end.yaml`（mm）。
- `4.test_gripper.py` 在观察位采集图像的同时用 `robot.get_T()` 记录当下 `T_end2base`，
  再做 `P_base = T_end2base @ T_camera2end @ P_camera`。

> 该 `T_camera2end.yaml` 是后续眼在手上 GraspNet 抓取的唯一标定依赖，务必先标好、用 ④ 验证通过。

---

## 3. 第二步：眼在手上 GraspNet 抓取

脚本 `graspnet-baseline/1.verify_grasp_eye_in_hand.py`，在 **`graspnet-baseline` 目录下**运行，
需：机械臂在线、Gemini 2 已接、GPU 可用、`checkpoint-rs.tar` 在当前目录、
`6d_eye_in_hand/T_camera2end.yaml` 已生成。
```bash
cd /home/yzxoi/Downloads/3d_grasp/graspnet-baseline
../.venv/bin/python 1.verify_grasp_eye_in_hand.py
```
open3d 窗口按键与眼在手外版相同：`W/S` 抬升高度、`A/D` 角度、`M` 依次尝试抓取（带逆解判断）、
`J/K` 夹爪开合、`I/U` 角度阈值、`O/P` 高度阈值、`Q` 退出。

与眼在手外版（`1.verify_grasp.py`）的差异：
- 读取 `../6d_eye_in_hand/T_camera2end.yaml`（相机->末端）。
- 每次取帧前先回到观察位，并用 `robot.get_T()` 记录采集瞬间的 `T_end2base`。
- `T_grasp2base = T_end2base @ T_camera2end @ T_grasp2camera`，再扣除舵机夹爪偏置 `T_servo2end`。
- 高度过滤用的「相机在基座系高度」按 `(T_end2base @ T_camera2end)[:3,3]` 实时计算
  （眼在手外是固定常量 `t_camera2base[2]`）。

> VLM 两进程脚本（`2.demo_VLM_grasp.py` + `3.demo_VLM_handler.py`）目前仅提供眼在手外版本。
> 若要改成眼在手上，套用同样的变换替换即可：加载 `T_camera2end`、采集时记录 `T_end2base`、
> `T_grasp2base = T_end2base @ T_camera2end @ T_grasp2camera`。

---

## 4. 坑（与眼在手外通用，务必了解）

- **深度单位**、**`torch.load(weights_only=False)`**、**`workspace_mask`**、
  **opencv 必须 4.10 contrib**、**机械臂须在线** 等注意事项与 `SETUP_GRASPNET.md` 第 4 节一致。
- 眼在手上额外注意：**观察位必须能让末端相机俯视整个工作区**，且采集图像与
  `get_T()` 必须在机械臂**静止于观察位**时进行（脚本已先 move 到观察位、sleep 后再采集）。
- 标定与抓取必须用**同一套相机安装**：相机相对末端一旦松动/移位，`T_camera2end` 失效需重标。
