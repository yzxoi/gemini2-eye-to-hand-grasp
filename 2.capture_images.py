# -*- coding: utf-8 -*-
"""
RealSense D435 采集标定图像（仅在检测到棋盘格时允许保存）

• 分辨率：1280 × 720 @ 30 FPS
• 按 <Space> 保存当前彩色帧（仅棋盘格成功检测时）
• 按 <ESC>/<q> 退出
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import time

# ─────────── 用户可调参数 ────────────
CHECKERBOARD     = (11, 8)          # 内角点 (columns, rows)
SQUARE_SIZE      = 20.4            # 单格边长 (mm)，仅保存深度时会用到
SAVE_DIR         = Path("./calib_imgs")
FNAME_TEMPLATE   = "rs_{:03d}.jpg"
SAVE_DEPTH_NPY   = False           # True → 同时保存深度为 .npy
# ────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)
print("保存路径:", SAVE_DIR.resolve())

# ───── RealSense 初始化 ─────
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 把深度帧“扭”进彩色相机的像素坐标系，使两张图一一对应。
align = rs.align(rs.stream.color)
pipeline.start(config)

# 打印分辨率信息
frames = align.process(pipeline.wait_for_frames())
h, w   = frames.get_color_frame().get_height(), frames.get_color_frame().get_width()
print(f"分辨率: {w}×{h}")

# 棋盘角点搜索参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_FAST_CHECK |
            cv2.CALIB_CB_NORMALIZE_IMAGE)

counter, t_prev = 0, time.perf_counter()
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    while True:
        # --------------- 采帧并对齐 ---------------
        frames       = align.process(pipeline.wait_for_frames())
        depth_frame  = frames.get_depth_frame()
        color_frame  = frames.get_color_frame()
        if not color_frame:
            continue
        color_img = np.asanyarray(color_frame.get_data())
        gray      = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # ---------- 棋盘格检测 ----------
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cb_flags)
        display_image = color_img.copy()
        if found:
            # 亚像素优化 + 角点可视化
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_image, CHECKERBOARD, corners, found)
            status_txt = "Chessboard detected  (SPACE=save)"
            txt_color  = (0, 255, 0)
        else:
            status_txt = "No chessboard  (adjust pose)"
            txt_color  = (0, 0, 255)

        # ---------- 叠加 FPS / 状态 ----------
        fps = 1.0 / (time.perf_counter() - t_prev)
        t_prev = time.perf_counter()
        cv2.putText(display_image, f"FPS: {fps:.1f}", (15, 40), font, 1, (0, 255, 0), 2)
        cv2.putText(display_image, status_txt, (15, h - 20), font, 0.8, txt_color, 2)

        cv2.imshow("D435 Capture", display_image)
        key = cv2.waitKey(1) & 0xFF

        # -------- 保存逻辑 --------
        if key == 32:                       # SPACE
            if found:
                fname = SAVE_DIR / FNAME_TEMPLATE.format(counter)
                cv2.imwrite(str(fname), color_img)
                print(f"[✓] Saved {fname.name}")

                if SAVE_DEPTH_NPY:
                    depth = np.asanyarray(depth_frame.get_data())
                    np.save(SAVE_DIR / f"{fname.stem}_depth.npy", depth)
                counter += 1
            else:
                print("[×] 未检测到棋盘格，未保存")

        # -------- 退出 --------
        if key in (27, ord('q')):           # ESC 或 q
            break

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    print(f"采集结束，共保存 {counter} 张有效标定图像")
