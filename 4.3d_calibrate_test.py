# ---------- 导入模块 ----------
import sys
import os
import argparse
import logging
import time
import random

import numpy as np
import cv2
import cv2.aruco as aruco

from episodeApp import EpisodeAPP
from orbbec_utils import OrbbecCamera

# ========== 生成标定点 ==========
def generate_cali_points(xy_offsets, z_vals):
    # 返回所有 (x, y, z) 组合
    return [(x, y, z) for z in z_vals for (x, y) in xy_offsets]

class Calibration:
    # ========== 初始化 ==========
    def __init__(self, robot_ip: str, robot_port: int, visualize: bool = False):
        # 日志设置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        # 参数
        self.sucker_length = 60      # mm
        self.marker_size = 0.05      # m
        self.xy_offsets = [
            (350, 0), (300, 50), (250, 0), (300, -50),
            (400, 0), (300, 100), (250, 0), (300, -100),
            (430, 0), (300, 150), (250, 0), (300, -150),
            (430, 0), (300, 200), (250, 0), (299, -200)
        ]
        self.z_vals = [10, 30, 50, 70, 90, 110]
        self.cali_points = generate_cali_points(self.xy_offsets, self.z_vals)
        self.visualize = visualize

        # 机器人初始化
        self.robot = EpisodeAPP(ip=robot_ip, port=robot_port)
        self.robot.move_xyz_rotation(
            [320, 0, 100], [180, 0, 90], rotation_order="xyz", speed_ratio=1
        )

        # Orbbec Gemini 2 相机流（深度软件对齐到彩色坐标系）
        self.camera = OrbbecCamera(color_w=1280, color_h=720, fps=30)

        # 可视化窗口
        if self.visualize:
            cv2.namedWindow("实时视图", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("实时视图", 2560, 720)

        # ArUco 设置
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters()

    # ========== 释放资源 ==========
    def cleanup(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        self.robot.gripper_off()
        self.logger.info("已释放所有资源。")

    # ========== 检测 ArUco ==========
    def get_aruco_center(self):
        color_image, depth_u16, depth_m = self.camera.wait_frames()
        if color_image is None:
            return None, None

        cam_matrix = self.camera.cam_matrix
        dist = self.camera.dist

        corners, ids, _ = aruco.detectMarkers(color_image, self.dictionary, parameters=self.parameters)
        center_point = None
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, cam_matrix, dist)
            aruco.drawDetectedMarkers(color_image, corners)
            for rvec, tvec, corner in zip(rvecs, tvecs, corners):
                cv2.drawFrameAxes(color_image, cam_matrix, dist, rvec, tvec, self.marker_size)
                x = float((corner[0][0][0] + corner[0][2][0]) / 2)
                y = float((corner[0][0][1] + corner[0][2][1]) / 2)
                d = self.camera.get_distance(depth_m, x, y)
                if d <= 0:
                    # 深度无效（空洞/越界），跳过本帧，避免污染标定
                    break
                xyz = self.camera.deproject(x, y, d)
                center_point = list(xyz)
                cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                txt = f"x:{xyz[0]:.3f} y:{xyz[1]:.3f} z:{xyz[2]:.3f}"
                cv2.putText(color_image, txt, (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                break

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_u16, alpha=0.14),
            cv2.COLORMAP_JET
        )
        combined = np.hstack((color_image, depth_colormap))

        if self.visualize:
            cv2.imshow("实时视图", combined)
            cv2.waitKey(1)

        return combined, center_point

    # ========== 执行标定 ==========
    def run_calibration(self):
        self.robot.gripper_on()
        self.logger.info("请固定 ArUco 标记于末端并保持吸附。")
        time.sleep(10)

        save_dir = "save_parms"
        os.makedirs(save_dir, exist_ok=True)
        cam2base = os.path.join(save_dir, "camera2base.npy")
        base2cam = os.path.join(save_dir, "base2camera.npy")

        if os.path.exists(cam2base) and os.path.exists(base2cam):
            self.T_camera2base = np.load(cam2base)
            self.T_base2camera = np.load(base2cam)
            self.logger.info("已加载现有变换矩阵。")
            return

        n = len(self.cali_points)
        base_coords = np.ones((4, n))
        cam_coords = np.ones((4, n))

        for i, (x, y, z) in enumerate(self.cali_points):
            tgt = [x, y, z + self.sucker_length]
            self.logger.info(f"标定点{i}：移动至 X={x},Y={y},Z={z}")
            self.robot.move_xyz_rotation(tgt, [180,0,90], rotation_order="xyz", speed_ratio=1)
            base_coords[:3, i] = [x+50, y, z]
            time.sleep(1)
            _, ctr = self.get_aruco_center()
            if ctr is None:
                self.logger.warning(f"标定点 {i} 未检测到标记，需要调整深度相机后重新标定。")
                return
            cam_coords[:3, i] = ctr

        self.T_camera2base = base_coords @ np.linalg.pinv(cam_coords)
        self.T_base2camera = np.linalg.pinv(self.T_camera2base)
        np.save(cam2base, self.T_camera2base)
        np.save(base2cam, self.T_base2camera)
        self.logger.info("标定完成并保存变换矩阵。")

    # ========== 执行识别抓取 ==========
    def run_recog(self):
        cam2base = os.path.join("save_parms", "camera2base.npy")
        if not os.path.exists(cam2base):
            print("T_camera2base 标定数据不存在")
            return
        if not hasattr(self, 'T_camera2base'):
            self.T_camera2base = np.load(cam2base)

        self.robot.gripper_off()
        time.sleep(1)

        try:
            while True:
                _, ctr = self.get_aruco_center()
                if ctr is None:
                    continue
                cam_pt = np.array([*ctr, 1.0])
                base_pt = self.T_camera2base @ cam_pt
                app = [base_pt[0], base_pt[1], base_pt[2] + self.sucker_length + 100]
                pk = [base_pt[0], base_pt[1], base_pt[2] + self.sucker_length - 8]
                self.robot.move_xyz_rotation(app, [180,0,90], rotation_order="xyz", speed_ratio=1)
                self.robot.move_xyz_rotation(pk, [180,0,90], rotation_order="xyz", speed_ratio=1)
                self.robot.gripper_on()
                self.robot.move_xyz_rotation(app, [180,0,90], rotation_order="xyz", speed_ratio=1)
                dx, dy = random.randint(250,380), random.randint(-110,210)
                drop = [dx, dy, self.sucker_length + 100]
                self.robot.move_xyz_rotation(drop, [180,0,90], rotation_order="xyz", speed_ratio=1)
                self.robot.gripper_off()
                time.sleep(1)
                self.logger.info("完成放置，寻找下一个目标。")
        except KeyboardInterrupt:
            self.logger.info("识别抓取终止。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机与机器人标定及识别脚本")
    parser.add_argument("--ip", default="localhost", help="机器人 IP 地址")
    parser.add_argument("--port", type=int, default=12345, help="机器人端口号")
    parser.add_argument("--visualize", action="store_true", help="启用可视化窗口显示")
    parser.add_argument("--calibrate", action="store_true", help="仅执行标定")
    parser.add_argument("--recognize", action="store_true", help="仅执行识别抓取")
    args = parser.parse_args()

    cal = Calibration(robot_ip=args.ip, robot_port=args.port, visualize=args.visualize)
    try:
        if args.calibrate:
            cal.run_calibration()
        if args.recognize:
            cal.run_recog()
        if not args.calibrate and not args.recognize:
            cal.run_calibration()
            cal.run_recog()
    finally:
        cal.cleanup()
