# -*- coding: utf-8 -*-
"""
使用 Orbbec Gemini 2 相机和 ArUco 标记进行物体抓取测试（眼在手外）

• 眼在手外：相机固定在外部，使用固定的 T_camera2base 将相机坐标直接转到基座坐标
• 使用 ArUco 标记进行物体定位
• 支持自动抓取和放置功能
• 用于验证 3.calibrate.py 标定出的 T_camera2base 是否准确

使用方法（在 6d 目录下运行）:
    ../.venv/bin/python 4.test_gripper.py  # 运行抓取测试程序
"""

import sys
import os
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import random
from episodeApp import EpisodeAPP
import yaml

# orbbec_utils 位于上一级目录（3d_grasp 根目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbbec_utils import OrbbecCamera


class Calibration:
    def __init__(self):
        """
        初始化抓取测试类
        """
        # ───── 基本参数设置 ─────
        self.sucker_length = 60  # 吸嘴长度（单位：mm）
        self.robot = EpisodeAPP()
        self.marker_size = 0.05  # ArUco标记尺寸（单位：m）
        self.last_drop_position = None  # 记录上一次放置位置

        # ───── Orbbec Gemini 2 初始化（深度对齐到彩色）─────
        self.camera = OrbbecCamera(color_w=1280, color_h=720, fps=30)

        # ───── ArUco 设置 ─────
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters()

    def get_aruco_center(self, calib=True):
        """
        检测 ArUco 标记并获取中心点坐标
        
        Args:
            calib (bool): 标定模式标志
            
        Returns:
            tuple: (图像, 中心点坐标)
                - 图像: 包含颜色帧和深度帧的组合图像
                - 中心点坐标: 如检测到则返回[x, y, z]列表，否则为None
        """
        # 获取相机帧（深度已对齐到彩色）
        color_image, depth_u16, depth_m = self.camera.wait_frames()
        if color_image is None:
            return None, None

        # 获取相机内参（畸变取 0，针孔模型）
        cam_matrix = self.camera.cam_matrix
        dist = self.camera.dist

        # 检测 ArUco 标记
        detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        corners, ids, rejected = detector.detectMarkers(color_image)
        
        center = None
        # 如果检测到标记
        if ids is not None and len(ids) > 0:
            # 估计标记位姿
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, cam_matrix, dist)
            # 绘制标记边框
            aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # 处理第一个检测到的标记
            for rvec, tvec, corner in zip(rvecs, tvecs, corners):
                # 绘制坐标轴
                cv2.drawFrameAxes(color_image, cam_matrix, dist, rvec, tvec, self.marker_size)
                
                # 计算标记中心点
                x = float((corner[0][0][0] + corner[0][2][0]) / 2)
                y = float((corner[0][0][1] + corner[0][2][1]) / 2)
                
                # 获取深度信息（米）
                d = self.camera.get_distance(depth_m, x, y)
                # 转换为相机坐标系下的3D坐标（米）
                xyz = self.camera.deproject(x, y, d)
                center = list(xyz)
                
                # 在图像上标记中心点
                cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # 在图像上显示坐标信息
                txt = f"x:{xyz[0]:.3f} y:{xyz[1]:.3f} z:{xyz[2]:.3f}"
                cv2.putText(color_image, txt, (int(x)+5, int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 只处理第一个标记
                break

        # 处理深度图像
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_u16, alpha=0.14),
            cv2.COLORMAP_JET
        )
        
        # 组合图像
        images = np.hstack((color_image, depth_colormap))
        
        return images, center
        
    def run_recog(self):
        """
        运行物体识别和抓取主循环
        """
        # ───── 加载标定参数（眼在手外：相机->基座，固定不变）─────
        with open('./T_camera2base.yaml', 'r') as f:
            T_data = yaml.safe_load(f)
            T_camera2base = np.array(T_data['T_camera2base'])

        print("相机到基座的变换矩阵：")
        print(T_camera2base)

        # ───── 初始化机器人状态 ─────
        self.robot.gripper_off()
        # 移动到初始位置（相机固定，把机械臂移开以免遮挡相机视野）
        self.robot.move_xyz_rotation([260, 0, 400], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
        time.sleep(1)
        
        # ───── 主循环 ─────
        while True:
            # 移动到观察位置
            self.robot.move_xyz_rotation([260, 0, 400], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
            time.sleep(1)

            # 检测标记
            images, center = self.get_aruco_center(calib=False)
            if center is not None:
                center = np.array(center) * 1000  # 转换为毫米
                
                cv2.imshow("image", images)
                cv2.waitKey(1)
                
                # ───── 坐标转换（眼在手外：相机->基座 一步到位）─────
                # 构建相机坐标系下的齐次坐标（单位：mm）
                P_camera = np.ones(4)
                P_camera[0:3] = center
                print(f'相对于相机的坐标：{P_camera[:3]}')

                # 直接转换到机器人基座坐标系
                P_base = T_camera2base @ P_camera
                print(f'相对于基座的坐标：{P_base[:3]}')
                print("--------------------------------")

                
                # 移动到物体上方
                self.robot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.sucker_length + 100], 
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )

                # ───── 执行抓取动作 ─────
                self.robot.gripper_on()
                
                # 移动到抓取位置
                self.robot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.sucker_length-15], 
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )

                # 向上移动20mm
                self.robot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.sucker_length + 20], 
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )
                
                # ───── 随机放置 ─────
                # 生成随机放置位置，确保与上一次放置位置有足够距离
                min_distance = 150  # 最小距离要求（毫米）
                while True:
                    dx, dy = random.randint(250,380), random.randint(-220,220)
                    if self.last_drop_position is None:
                        break
                    # 计算与上一次放置位置的距离
                    distance = np.sqrt((dx - self.last_drop_position[0])**2 + 
                                     (dy - self.last_drop_position[1])**2)
                    if distance >= min_distance:
                        break
                
                drop = [dx, dy, self.sucker_length + 100]
                self.last_drop_position = [dx, dy]  # 更新上一次放置位置
                self.robot.move_xyz_rotation(
                    drop,
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )
                
                # 释放物体
                self.robot.gripper_off()
                time.sleep(1)
                
            else:
                print("no marker detected")
                time.sleep(1)
                
    def cleanup(self):
        """
        清理资源：停止相机流、关闭窗口、关闭夹爪
        """
        self.camera.stop()
        cv2.destroyAllWindows()
        self.robot.gripper_off()
        print("已释放所有资源。")

# 主程序入口
if __name__ == "__main__":
    cali = Calibration()
    try:
        cali.run_recog()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        cali.cleanup()