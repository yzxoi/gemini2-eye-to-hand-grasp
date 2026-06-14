# -*- coding: utf-8 -*-
"""
Orbbec Gemini 2 相机工具模块
=================================
封装 pyorbbecsdk，提供一个与原 RealSense 代码风格相近的简单接口，
用于替换 4.3d_calibrate_test.py / 1.test_d435_aruco.py 中的 pyrealsense2。

依赖：
    pip/源码安装 pyorbbecsdk（需与已安装的 OrbbecSDK 版本匹配）
    numpy, opencv-python

说明：
    - 同时兼容 pyorbbecsdk v1.x 与 v2.x 的高层 Python API。
    - 彩色帧会被解码为 OpenCV 使用的 BGR 图像。
    - 深度帧通过 AlignFilter 软件对齐到彩色帧坐标系（D2C），
      因此反投影使用的是彩色相机内参。
    - get_distance() 返回单位为「米」，与 RealSense depth_frame.get_distance() 保持一致，
      这样标定与抓取使用同一坐标尺度即可，绝对单位不影响线性最小二乘标定结果。
"""

import numpy as np
import cv2

from pyorbbecsdk import (
    Pipeline,
    Config,
    AlignFilter,
    Context,
    OBSensorType,
    OBStreamType,
    OBFormat,
    OBAlignMode,
    OBLogLevel,
    OBError,
)


# Gemini 2 固件会周期性上报 "Timestamp anomaly detected" 的非致命错误日志，
# 不影响取流。这里把控制台日志级别调高以减少刷屏（仅保留 FATAL）。
def quiet_sdk_log(level=OBLogLevel.FATAL):
    try:
        Context.set_logger_to_console(level)
    except Exception:
        pass


quiet_sdk_log()


# ---------------------------------------------------------------------------
# 彩色帧 -> BGR 图像
# ---------------------------------------------------------------------------
def _yuyv_to_bgr(data, w, h):
    yuyv = data.reshape((h, w, 2))
    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)


def _uyvy_to_bgr(data, w, h):
    uyvy = data.reshape((h, w, 2))
    return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)


def _i420_to_bgr(data, w, h):
    yuv = data.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def _nv12_to_bgr(data, w, h):
    yuv = data.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def _nv21_to_bgr(data, w, h):
    yuv = data.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def frame_to_bgr_image(color_frame):
    """把 Orbbec 彩色帧转换为 OpenCV BGR ndarray，失败返回 None。"""
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if fmt == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        image = data.reshape((height, width, 3))
    elif fmt == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.YUYV:
        image = _yuyv_to_bgr(data, width, height)
    elif fmt == OBFormat.UYVY:
        image = _uyvy_to_bgr(data, width, height)
    elif fmt == OBFormat.I420:
        image = _i420_to_bgr(data, width, height)
    elif fmt == OBFormat.NV12:
        image = _nv12_to_bgr(data, width, height)
    elif fmt == OBFormat.NV21:
        image = _nv21_to_bgr(data, width, height)
    else:
        print(f"[orbbec_utils] 不支持的彩色格式: {fmt}")
        return None
    return image


# ---------------------------------------------------------------------------
# 相机封装
# ---------------------------------------------------------------------------
class OrbbecCamera:
    """
    简单的 Gemini 2 相机封装。

    用法：
        cam = OrbbecCamera(color_w=1280, color_h=720, fps=30)
        color_bgr, depth_u16, depth_m = cam.wait_frames()
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        xyz = cam.deproject(u, v, depth_m[v, u])  # 单位：米
        cam.stop()
    """

    def __init__(self, color_w=1280, color_h=720, fps=30, enable_color_sync=True):
        self.pipeline = Pipeline()
        config = Config()

        # ---- 彩色流：优先 RGB，便于直接解码；失败则回退到默认 profile ----
        color_profile = None
        try:
            color_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            for fmt in (OBFormat.RGB, OBFormat.MJPG, OBFormat.BGR):
                try:
                    color_profile = color_list.get_video_stream_profile(color_w, color_h, fmt, fps)
                    if color_profile is not None:
                        break
                except OBError:
                    continue
            if color_profile is None:
                color_profile = color_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
        except OBError as e:
            raise RuntimeError(f"无法启用彩色流: {e}")

        # ---- 深度流：尽量与彩色同分辨率，失败则用默认 ----
        try:
            depth_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = None
            try:
                depth_profile = depth_list.get_video_stream_profile(color_w, color_h, OBFormat.Y16, fps)
            except OBError:
                depth_profile = None
            if depth_profile is None:
                depth_profile = depth_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
        except OBError as e:
            raise RuntimeError(f"无法启用深度流: {e}")

        # 帧同步（彩色/深度时间对齐）
        if enable_color_sync:
            try:
                self.pipeline.enable_frame_sync()
            except Exception:
                pass

        # ---- 深度对齐到彩色坐标系（D2C）----
        # 优先用硬件 D2C（由相机芯片完成，省 CPU、帧率更高）；当前分辨率/格式
        # 组合不支持时回退到软件 AlignFilter（原行为）。
        self.align_filter = None
        self.hw_align = False
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
            self.pipeline.start(config)
            self.hw_align = True
        except OBError:
            # 硬件对齐不支持 -> 关闭后用软件对齐重新启动
            try:
                config.set_align_mode(OBAlignMode.DISABLE)
            except Exception:
                pass
            self.pipeline.start(config)
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        print(f"[orbbec_utils] D2C 对齐方式: {'硬件 HW' if self.hw_align else '软件 SW'}；丢弃积压旧帧: 开")

        # ---- 读取彩色相机内参（对齐到彩色后，反投影用彩色内参）----
        self.fx = self.fy = self.cx = self.cy = None
        # 畸变恒为 0：见 _load_intrinsics 说明（彩色图已矫正，反投影按针孔模型处理）
        self.dist = np.zeros(5, dtype=np.float64)
        self._load_intrinsics(color_profile)

    # ----------------------------------------------------------------
    def _load_intrinsics(self, color_profile):
        """
        优先用彩色 stream profile 的内参，回退到 pipeline.get_camera_param()。

        关于畸变：Gemini 2 上报的是 8 参数 rational 模型 (k1..k6, p1, p2)，
        实测 k1≈k4、k2≈k5、k3≈k6，分子≈分母 → 径向畸变系数 ≈ 1（画面最边角也仅约 0.5%），
        即彩色图本身基本已矫正。本工程与课程文档一致，反投影采用纯针孔模型，
        故 self.dist 统一取 0，避免「只取前 5 个 rational 系数」导致的严重误畸变。
        （若将来更换为有真实畸变的相机，应改为传完整 8 参数 [k1,k2,p1,p2,k3,k4,k5,k6]
          并在 deproject 中做去畸变。）
        """
        intr = None
        # 方式一：从 profile 直接取（分辨率天然匹配）
        try:
            vsp = color_profile.as_video_stream_profile()
            intr = vsp.get_intrinsic()
        except Exception:
            intr = None

        # 方式二：从 pipeline 取 RGB 内参
        if intr is None:
            try:
                param = self.pipeline.get_camera_param()
                intr = param.rgb_intrinsic
            except Exception as e:
                raise RuntimeError(f"无法获取相机内参: {e}")

        self.fx = float(intr.fx)
        self.fy = float(intr.fy)
        self.cx = float(intr.cx)
        self.cy = float(intr.cy)

    # ----------------------------------------------------------------
    @property
    def cam_matrix(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    # ----------------------------------------------------------------
    def wait_frames(self, timeout_ms=200, drain=True):
        """
        返回 (color_bgr, depth_u16, depth_m)：
            color_bgr : HxWx3 BGR 图像
            depth_u16 : HxW uint16 原始深度（单位由 depth_scale 决定，通常为 mm）
            depth_m   : HxW float32 深度（单位：米）
        某一帧缺失时返回 (None, None, None)。

        drain=True 时丢弃管线里积压的旧帧、只保留最新一帧。
        这很关键：抓取/标定时机械臂运动会阻塞主循环数秒，期间相机帧不断入队，
        若不丢弃，下一次取到的是「几秒前的旧画面」，导致程序看不到目标已被移动
        （实时状态滞后）。这里用非阻塞轮询把积压帧排空到最新。
        """
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None, None

        if drain:
            # 非阻塞地取走队列里已就绪的更新帧；队列空时 2ms 内返回 None 跳出。
            # cap 仅作安全上限，正常情况下队列很快排空。
            for _ in range(64):
                newer = self.pipeline.wait_for_frames(2)
                if newer is None:
                    break
                frames = newer

        # 软件对齐路径（硬件 D2C 时 self.align_filter 为 None，帧已对齐）
        if self.align_filter is not None:
            frames = self.align_filter.process(frames)
            if frames is None:
                return None, None, None
            # AlignFilter 输出需转回 FrameSet
            try:
                frames = frames.as_frame_set()
            except Exception:
                pass

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None, None

        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            return None, None, None

        h = depth_frame.get_height()
        w = depth_frame.get_width()
        scale = depth_frame.get_depth_scale()  # 单位/级 -> 毫米
        depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
        depth_m = depth_u16.astype(np.float32) * scale / 1000.0  # -> 米

        return color_bgr, depth_u16, depth_m

    # ----------------------------------------------------------------
    def get_distance(self, depth_m, u, v):
        """返回像素 (u, v) 的深度（米）；越界或无效返回 0.0。"""
        h, w = depth_m.shape[:2]
        u = int(round(u)); v = int(round(v))
        if u < 0 or v < 0 or u >= w or v >= h:
            return 0.0
        return float(depth_m[v, u])

    # ----------------------------------------------------------------
    def deproject(self, u, v, z):
        """像素 (u, v) + 深度 z（米）-> 相机坐标 [X, Y, Z]（米）。"""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return [float(x), float(y), float(z)]

    # ----------------------------------------------------------------
    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
