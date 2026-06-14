import glob
import numpy as np
import cv2

# ---------- 用户可修改的参数 ----------
# ① 棋盘格规格 (内角点个数)
CHECKERBOARD = (11, 8)            # (columns, rows)
# ② 单个方格的物理边长 (mm 或 m 皆可)
SQUARE_SIZE  = 20.4          # mm
# ③ 标定图片所在文件夹
IMG_PATH_GLOB = "./calib_imgs/*.jpg"
# -------------------------------------

# ========== 1. 准备棋盘格在世界坐标中的3D点 ==========
objp = np.zeros((CHECKERBOARD[1]*CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE                               # 乘物理尺寸

objpoints = []   # 3D points
imgpoints = []   # 2D points

# 终止条件：最大30次迭代或精度<0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# ========== 2. 遍历所有图片 ==========
for fname in glob.glob(IMG_PATH_GLOB):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # 亚像素优化
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        objpoints.append(objp)

        # 可视化检查
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('corner', img)
        cv2.waitKey(50)

cv2.destroyAllWindows()
print(f"共检测到 {len(objpoints)} 张有效图片")

# ========== 3. 标定 ==========
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)



print("重投影均方误差 (RMSE) :", ret)
print("相机内参矩阵 K :\n", K)
print("畸变系数 [k1,k2,p1,p2,k3] :\n", dist.ravel())

# ========== 4. 计算总体重投影误差 ==========
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("平均重投影误差 = ", total_error / len(objpoints))

# ========== 5. 保存参数 ==========
fs = cv2.FileStorage("camera_intrinsic.yml", cv2.FILE_STORAGE_WRITE)
fs.write("K", K)
fs.write("dist", dist)
fs.release()
print("参数已写入 camera_intrinsic.yml")