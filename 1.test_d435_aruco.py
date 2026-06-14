# -*- coding: utf-8 -*-
"""
Orbbec Gemini 2 + ArUco 5x5-100 实时检测测试
（原为 Intel RealSense D435，已移植到 Orbbec pyorbbecsdk）

用途：
    验证相机能否正常出图、深度是否有效、ArUco 标定板检测是否正常，
    以及相机视角能否覆盖机械臂预设标定点。
按 q 或 ESC 退出。
"""

import time
import numpy as np
import cv2
import cv2.aruco as aruco

from orbbec_utils import OrbbecCamera

# ───────────────  ArUco 相关对象只创建一次  ───────────────
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(dictionary, parameters)

# ───────────────  Orbbec 相机  ───────────────
cam = OrbbecCamera(color_w=1280, color_h=720, fps=30)
intr_matrix = cam.cam_matrix.astype(np.float32)
intr_coeffs = cam.dist.astype(np.float32)
print("彩色相机内参矩阵：")
print(intr_matrix)

try:
    prev_time = time.perf_counter()
    while True:
        # ───── 获取对齐后的彩色/深度帧 ─────
        color_img, depth_u16, depth_m = cam.wait_frames()
        if color_img is None:
            continue

        # ───── ArUco 检测 ─────
        corners, ids, _ = aruco_detector.detectMarkers(color_img)

        if ids is not None and len(ids):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.05, intr_matrix, intr_coeffs
            )

            aruco.drawDetectedMarkers(color_img, corners)
            cv2.drawFrameAxes(color_img, intr_matrix, intr_coeffs,
                              rvec[0], tvec[0], 0.05)

            for _id, corner in zip(ids.flatten(), corners):
                # 计算中心点
                x = int((corner[0][0][0] + corner[0][2][0]) / 2)
                y = int((corner[0][0][1] + corner[0][2][1]) / 2)
                cv2.circle(color_img, (x, y), 10, (0, 0, 255), -1)

                # 深度 → 相机坐标（手动反投影，单位：米）
                z = cam.get_distance(depth_m, x, y)
                x_cam, y_cam, z_cam = cam.deproject(x, y, z)

                # 终端打印
                print(f"id={_id} px=({x},{y}) cam=({x_cam:.4f}, {y_cam:.4f}, {z_cam:.4f}) m")

                # 画面文字
                txt = f"x:{x_cam:.3f}, y:{y_cam:.3f}, z:{z_cam:.3f}"
                cv2.putText(color_img, txt, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ───── 深度伪彩 + FPS ─────
        depth_viz = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_u16, alpha=0.2), cv2.COLORMAP_JET
        )

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(color_img, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Orbbec Gemini 2', np.hstack((color_img, depth_viz)))

        # ───── 退出判断 ─────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    cv2.destroyAllWindows()
    cam.stop()
