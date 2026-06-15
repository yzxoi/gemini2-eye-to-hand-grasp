# gemini2-eye-to-hand-grasp

Eye-to-hand 3D vision robotic grasping on an **Orbbec Gemini 2** depth camera + **Episode** robot arm:
ArUco & chessboard hand-eye calibration, Grounding DINO open-vocabulary detection, and
**GraspNet** 6-DoF grasping. Ported from Intel RealSense D435 and CUDA-accelerated (RTX-class GPU).

> **Eye-to-hand**: the camera is fixed in the world (not mounted on the arm). The camera→base
> transform `T_camera2base` is a fixed rigid matrix, used to map detected/grasp poses from the
> camera frame into the robot base frame.
>
> An **eye-in-hand** variant (camera mounted on the end-effector) is also included — see
> `6d_eye_in_hand/` and `graspnet-baseline/1.verify_grasp_eye_in_hand.py` (`SETUP_EYE_IN_HAND.md`).

## Hardware / stack
- Camera: Orbbec Gemini 2 (`pyorbbecsdk2`)
- Arm: Episode (`episodeApp.py`, TCP `localhost:12345`)
- GPU: NVIDIA (CUDA 13 / PyTorch cu130) for Grounding DINO + GraspNet
- Python 3.12 (`uv` venv)

## Layout
| Path | What |
|------|------|
| `orbbec_utils.py` | Gemini 2 wrapper (`OrbbecCamera`): color/depth, D2C alignment, deprojection |
| `1.test_d435_aruco.py` | Camera + ArUco sanity check |
| `4.3d_calibrate_test.py` | Simple ArUco eye-to-hand calibration + pick demo |
| `5.dino_detect.py`, `6.dino_grasp.py` | Grounding DINO detection / detect-and-grasp |
| `6d/` | Chessboard hand-eye calibration (eye-to-hand) → `T_camera2base` |
| `6d_eye_in_hand/` | Chessboard hand-eye calibration (eye-in-hand) → `T_camera2end` |
| `graspnet-baseline/` | GraspNet 6-DoF grasping (`1.verify_grasp.py` + eye-in-hand variant, VLM pipeline `2`+`3`) |
| `SETUP_ORBBEC.md` | Camera setup, env, ArUco calibration, 6d workflow |
| `SETUP_GRASPNET.md` | GraspNet env/build, eye-to-hand calibration → grasp workflow, gotchas |
| `SETUP_EYE_IN_HAND.md` | Eye-in-hand calibration → grasp workflow (camera on the arm) |

## Quick start
See **`SETUP_ORBBEC.md`** (camera + calibration) and **`SETUP_GRASPNET.md`** (grasping).
High level:
1. Set up the `uv` venv and `pyorbbecsdk2` (SETUP_ORBBEC.md).
2. Run the `6d/` chessboard hand-eye calibration → `6d/T_camera2base.yaml`.
3. Run grasping: `graspnet-baseline/1.verify_grasp.py` (point cloud) or the VLM pipeline.

## Not included in this repo (download separately)
- Model weights: Grounding DINO (`grounding-dino-base/`) and GraspNet checkpoint
  (`graspnet-baseline/checkpoint-rs.tar`).
- The Python venv, compiled CUDA extensions, and machine-specific calibration outputs
  (`save_parms/`, `6d/T_camera2base.yaml`) — regenerate per machine (see the setup docs).
