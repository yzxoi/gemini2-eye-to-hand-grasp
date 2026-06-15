# -*- coding: utf-8 -*-
"""
使用 Orbbec Gemini 2 相机和棋盘格进行手眼标定（眼在手上 / eye-in-hand）

• 分辨率：1280 × 720 @ 30 FPS
• 眼在手上：相机固定在机械臂末端，随臂运动；棋盘格固定在工作台（基座系不动）
• 使用棋盘格估计每张图的 board->camera 位姿，结合 end->base 做手眼标定
• 支持多种手眼标定算法：HORAUD、TSAI、PARK
• 输出「相机到末端」的刚体变换 T_camera2end.yaml（单位：mm）

使用方法（在 6d_eye_in_hand 目录下运行）:
    ../.venv/bin/python 3.calibrate.py  # 运行标定程序

与眼在手外（6d/）的唯一区别：
    OpenCV calibrateHandEye 原生求解的就是眼在手上（cam2gripper），因此**直接**
    把「末端->基座」位姿喂进去即可，无需取逆；返回的 X 即「相机->末端」T_camera2end。
"""

import sys
import cv2
import numpy as np
import glob
import os
import yaml

# orbbec_utils 位于上一级目录（3d_grasp 根目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbbec_utils import OrbbecCamera

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()

class Calibration:
    def __init__(self, pattern_size=tuple(config['checkerboard']['pattern_size']),
                 square_size=config['checkerboard']['square_size']):
        """
        初始化标定类
        :param pattern_size: 棋盘格内角点数量 (columns, rows)
        :param square_size: 棋盘格方格尺寸 (mm)
        """
        self.pattern_size = pattern_size
        self.square_size = square_size

        # 存储中间结果
        self.obj_points = []
        self.img_points = []
        self.rvecs = []
        self.t_board2cameras = []
        self.intr_matrix = None
        self.dist_coeffs = None
        self.R_camera2end = None
        self.t_camera2end = None
        self.T_camera2end = None

        # 棋盘格角点检测参数
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        self.cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                        cv2.CALIB_CB_FAST_CHECK |
                        cv2.CALIB_CB_NORMALIZE_IMAGE)

        # ───── Orbbec Gemini 2 初始化 ─────
        self.camera = OrbbecCamera(color_w=config['camera']['resolution'][0],
                                   color_h=config['camera']['resolution'][1],
                                   fps=config['camera']['fps'])

    def get_camera_paras(self, calib=True):
        """
        获取相机内参（Orbbec 出厂内参；Gemini 2 彩色图已基本矫正，畸变取 0，针孔模型）
        :return: 相机内参矩阵和畸变系数
        """
        return self.camera.cam_matrix, self.camera.dist

    @staticmethod
    def assess_image_quality(image):
        """
        评估图像质量
        :param image: 输入图像
        :return: 图像质量分数（拉普拉斯方差）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def calculate_reprojection_error(self, obj_points, img_points, rvecs, tvecs, intr_matrix, dist_coeffs):
        """
        计算重投影误差以评估标定质量
        :param obj_points: 3D点坐标
        :param img_points: 2D图像点坐标
        :param rvecs: 旋转向量
        :param tvecs: 平移向量
        :param intr_matrix: 相机内参矩阵
        :param dist_coeffs: 畸变系数
        :return: 平均重投影误差和各视图误差
        """
        total_error = 0
        total_points = 0
        per_view_errors = []

        for i in range(len(obj_points)):
            # 使用标定参数将3D点投影到图像平面
            projected_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], intr_matrix, dist_coeffs)

            # 计算投影点与实际检测到的角点之间的误差
            error = cv2.norm(img_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            per_view_errors.append(error)
            total_error += error
            total_points += len(obj_points[i])

        # 平均重投影误差
        mean_error = total_error / len(obj_points)

        return mean_error, per_view_errors

    def run_calibration(self, image_path, method=cv2.CALIB_HAND_EYE_HORAUD):
        """
        运行手眼标定
        :param image_path: 标定图像路径
        :param method: 手眼标定算法
        """
        # 重置存储的中间结果
        self.obj_points = []
        self.img_points = []
        self.rvecs = []
        self.t_board2cameras = []

        # 读取并排序标定图像
        images = glob.glob(os.path.join(image_path, '*.jpg'))
        images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        print(f"共找到 {len(images)} 张图像。")

        # 准备标定点
        obj_points = []
        img_points = []
        objp = np.zeros((np.prod(self.pattern_size), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size

        det_success_num = 0
        low_quality_images = []

        # 处理每张标定图像
        for i, image in enumerate(images):
            img = cv2.imread(image)
            if img is None:
                print(f"无法读取图像: {image}")
                continue

            # 评估图像质量
            quality_score = self.assess_image_quality(img)
            print(f"图像 {image} 的质量分数: {quality_score}")

            if quality_score < config['image_quality']['calibration_threshold']:
                low_quality_images.append(image)
                cv2.putText(img, "Low Quality", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Low Quality Image', img)
                cv2.waitKey(1000)
            else:
                # 检测棋盘格角点
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, self.cb_flags)
                if ret:
                    det_success_num += 1
                    # 亚像素优化 + 角点可视化
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    obj_points.append(objp)
                    img_points.append(corners2)
                    cv2.drawChessboardCorners(img, self.pattern_size, corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(100)

        cv2.destroyAllWindows()

        # 获取相机内参
        self.intr_matrix, self.dist_coeffs = self.get_camera_paras()
        print("相机内参矩阵: ")
        print(self.intr_matrix)

        # 计算每张图像的位姿
        R_board2cameras = []
        t_board2cameras = []
        rvecs = []

        for i in range(det_success_num):
            ret, rvec, t_board2camera = cv2.solvePnP(obj_points[i], img_points[i], self.intr_matrix, self.dist_coeffs)
            rvecs.append(rvec)
            R_board2camera, _ = cv2.Rodrigues(rvec)
            R_board2cameras.append(R_board2camera)
            t_board2cameras.append(t_board2camera)

        # 保存中间结果
        self.obj_points = obj_points
        self.img_points = img_points
        self.rvecs = rvecs
        self.t_board2cameras = t_board2cameras

        # 计算重投影误差
        mean_error, per_view_errors = self.calculate_reprojection_error(
            obj_points, img_points, rvecs, t_board2cameras, self.intr_matrix, self.dist_coeffs)

        # 输出标定质量评估结果
        print("\n=== 标定质量评估 ===")
        print(f"平均重投影误差: {mean_error:.6f} 像素")
        print("各视图重投影误差:")
        for i, error in enumerate(per_view_errors):
            print(f"  图像 {i+1}: {error:.6f} 像素")

        # 重投影误差评估标准
        if mean_error < 0.5:
            print("标定质量: 非常好 (误差 < 0.5 像素)")
        elif mean_error < 1.0:
            print("标定质量: 良好 (误差 < 1.0 像素)")
        elif mean_error < 1.5:
            print("标定质量: 一般 (误差 < 1.5 像素)")
        else:
            print("标定质量: 较差 (误差 >= 1.5 像素)，建议重新标定")

        # 加载机械臂位姿数据（末端到基座 T_end2base）
        with open(f"{config['paths']['save_dir']}/T_end2base.yaml", 'r') as f:
            T_data = yaml.safe_load(f)
            T_list = np.array(T_data['T_end2base'])

        # 眼在手上（eye-in-hand）：相机固定在末端，棋盘格固定在基座系。
        # OpenCV calibrateHandEye 原生求解的就是眼在手上(cam2gripper)，因此直接把
        # 「末端->基座」位姿喂进去即可（无需取逆），返回的 X 即「相机->末端」cam2end。
        R_end2bases = []
        t_end2bases = []
        for T_end2base in T_list:
            R_end2bases.append(T_end2base[:3, :3])
            t_end2bases.append(T_end2base[:3, 3])

        # 执行手眼标定（眼在手上，得到相机 -> 末端）
        self.R_camera2end, self.t_camera2end = cv2.calibrateHandEye(
            R_end2bases, t_end2bases,
            R_board2cameras, t_board2cameras,
            method=method
        )

        # 构建齐次变换矩阵
        self.T_camera2end = np.eye(4)
        self.T_camera2end[:3, :3] = self.R_camera2end
        self.T_camera2end[:3, 3] = self.t_camera2end.reshape(3)

        np.set_printoptions(suppress=True, precision=10)

        # 输出标定结果
        print("\n=== 手眼标定结果（眼在手上）===")
        print("相机到末端的旋转矩阵:")
        print(self.R_camera2end)
        print("相机到末端的位移向量:")
        print(self.t_camera2end, "（单位：mm）")

        # 保存标定结果（相机 -> 末端 刚体变换，单位 mm）
        T_data = {'T_camera2end': self.T_camera2end.tolist()}
        with open('./T_camera2end.yaml', 'w') as f:
            yaml.dump(T_data, f)

        return mean_error

# 主程序入口
if __name__ == "__main__":
    # 初始化标定类
    cali = Calibration(pattern_size=tuple(config['checkerboard']['pattern_size']),
                      square_size=config['checkerboard']['square_size'])

    # 可选的手眼标定算法
    methods = {
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,  # Horaud方法
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,      # Tsai方法
        "Park": cv2.CALIB_HAND_EYE_PARK       # Park方法
    }

    print("\n=== 手眼标定方法选择 ===")
    print("可选的方法:")
    for i, name in enumerate(methods.keys(), 1):
        print(f"{i}. {name}")

    while True:
        try:
            choice = int(input("\n请选择标定方法 (输入数字 1-3): "))
            if 1 <= choice <= len(methods):
                method_name = list(methods.keys())[choice-1]
                method = methods[method_name]
                break
            else:
                print("无效的选择，请重新输入")
        except ValueError:
            print("请输入有效的数字")

    print(f"\n使用 {method_name} 方法进行标定...")

    # 运行标定
    mean_error = cali.run_calibration(config['paths']['save_dir'], method)

    # 输出标定结果
    print("\n=== 标定结果（眼在手上）===")
    print(f"相机内参标定质量 (重投影误差): {mean_error:.6f} 像素")
    print("\n相机到末端的旋转矩阵:")
    print(cali.R_camera2end)
    print("相机到末端的位移向量:")
    print(cali.t_camera2end, "（单位：mm）")
    print("相机到末端的齐次变换矩阵:")
    print(cali.T_camera2end)

    # 保存标定结果（相机 -> 末端）
    T_data = {'T_camera2end': cali.T_camera2end.tolist()}
    with open('./T_camera2end.yaml', 'w') as f:
        yaml.dump(T_data, f)
