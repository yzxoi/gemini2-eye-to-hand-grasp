# ---------- 环境与模块配置 ----------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ctypes
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Value, Manager
import threading
from orbbec_utils import OrbbecCamera
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from episodeApp import EpisodeAPP
import argparse

# ========== 零样本检测与ROI交互 ==========
class GraspAgent:
    def __init__(self, robot_ip, robot_port, classes, messages, running):
        # 状态与参数
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.classes = classes      # 共享目标类列表
        self.messages = messages    # 共享消息列表
        self.running = running      # 运行标志
        self.messages[0] = "Press 'R' to draw ROI"
        self.center_p_queue = Queue()
        self.status = Value(ctypes.c_int8, 0)  # 0: READY, 1: WORKING
        self.select_roi = False
        self.roi_points = []
        self.roi = None
        self.scale = 0.7
        self.frame_w, self.frame_h = 1280, 720
        # 模型只在检测子进程内加载一次（懒加载），避免逐帧重复加载 ~700MB 模型
        self._processor = None
        self._model = None

    # ---------- 目标检测 ----------
    def grounding_dino(self, image, classes):
        # 首次调用时加载模型并缓存（在检测子进程内）
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained("./grounding-dino-base")
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "./grounding-dino-base"
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        processor, model = self._processor, self._model
        text = "".join(f"{c}." for c in classes)
        image_pil = Image.fromarray(image[:, :, ::-1])  # BGR->RGB
        inputs = processor(images=image_pil, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=0.4, text_threshold=0.4,
            target_sizes=[image_pil.size[::-1]]
        )

    # ---------- ROI 绘制回调 ----------
    def mouse_callback(self, event, x_disp, y_disp, flags, _):
        if self.select_roi and event == cv2.EVENT_LBUTTONDOWN:
            x = min(int(x_disp / self.scale), self.frame_w - 1)
            y = min(int(y_disp / self.scale), self.frame_h - 1)
            self.roi_points.append((x, y))
            self.messages[0] = f"Point {len(self.roi_points)}: ({x}, {y})"
            if len(self.roi_points) == 2:
                (x1, y1), (x2, y2) = self.roi_points
                self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self.select_roi = False
                self.messages[0] = "ROI set: Enter classes"

    # ---------- 实时检测视频流 ----------
    def realsense_video(self):
        cam = OrbbecCamera(color_w=self.frame_w, color_h=self.frame_h, fps=30)
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detection', self.mouse_callback)
        try:
            while self.running.value:
                t0 = time.time()
                color, depth_u16, depth_m = cam.wait_frames()
                if color is None:
                    continue

                # 绘制ROI
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cls = list(self.classes)
                detected = False
                if cls:
                    img = color
                    off = (0, 0)
                    if self.roi:
                        x1, y1, x2, y2 = self.roi
                        img = color[y1:y2, x1:x2]
                        off = (x1, y1)
                    results = self.grounding_dino(img, cls)
                    for res in results:
                        for box, label, score in zip(res['boxes'], res['text_labels'], res['scores']):
                            l, t, r2, b2 = map(int, box)
                            l += off[0]; r2 += off[0]; t += off[1]; b2 += off[1]
                            cv2.rectangle(color, (l, t), (r2, b2), (0, 255, 0), 2)
                            cv2.putText(color, f"{label} {score:.2f}", (l, t-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cx, cy = (l + r2) // 2, (t + b2) // 2
                            dist = cam.get_distance(depth_m, cx, cy)
                            xyz = cam.deproject(cx, cy, dist)
                            cv2.circle(color, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(color, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}",
                                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            while not self.center_p_queue.empty():
                                try: self.center_p_queue.get_nowait()
                                except: break
                            self.center_p_queue.put(xyz)
                            detected = True
                    if not detected:
                        self.messages[0] = f"Classes: {cls} Not detected"

                # 信息叠加
                fps = 1.0 / (time.time() - t0)
                cv2.putText(color, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                state = "WORKING" if self.status.value else "READY"
                cv2.putText(color, f"ROBOT: {state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255) if self.status.value else (0, 255, 0), 2)
                cv2.putText(color, f"STATUS: {self.messages[0]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

                # 深度与彩色拼接
                dcol = cv2.applyColorMap(cv2.convertScaleAbs(depth_u16, alpha=0.14), cv2.COLORMAP_JET)
                combined = np.hstack((color, dcol))
                disp = cv2.resize(combined, (0, 0), fx=self.scale, fy=self.scale)
                cv2.imshow('Detection', disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running.value = False
                    break
                if key == ord('r'):
                    self.select_roi = True; self.roi_points = []; self.roi = None
                    self.messages[0] = "Press 'R' then click two points"
                if key == ord('c'):
                    self.roi = None; self.messages[0] = "Press 'R' to draw ROI"
        finally:
            cam.stop(); cv2.destroyAllWindows()

    # ---------- 机器人抓取 ----------
    def episode_robot_grasp(self):
        path = os.path.join("./save_parms", "camera2base.npy")
        if not os.path.exists(path):
            self.messages[0] = "Error: cannot load transform"
            print("Error: cannot load transform")
            self.running.value = False
            return
        T = np.load(path)
        suck = 60
        robot = EpisodeAPP(ip=self.robot_ip, port=self.robot_port)
        while self.running.value:
            cls = list(self.classes)
            if cls:
                self.messages[0] = f"Classes: {cls} waiting detection"
                while not self.center_p_queue.empty():
                    try: self.center_p_queue.get_nowait()
                    except: break
                try:
                    pt = self.center_p_queue.get(timeout=1)
                except:
                    continue
                time.sleep(1)
                if pt and self.running.value:
                    self.status.value = 1
                    self.messages[0] = f"Classes: {cls} grasping"
                    p = np.ones(4); p[:3] = pt
                    wp = T @ p
                    approach = [wp[0], wp[1], wp[2] + suck + 100]
                    pick = [wp[0], wp[1], wp[2] + suck]
                    robot.move_xyz_rotation(approach, [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.move_xyz_rotation(pick, [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.gripper_on()
                    robot.move_xyz_rotation(approach, [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.move_xyz_rotation([140, -300, 300], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
                    robot.gripper_off()
                    self.messages[0] = "Input classes"
                    self.classes[:] = []
                    self.status.value = 0
            time.sleep(0.1)

    # ---------- 主流程 ----------
    def run(self):
        p1 = Process(target=self.realsense_video)
        p2 = Process(target=self.episode_robot_grasp)
        p1.start(); p2.start()
        p1.join(); p2.join()

# ========== 脚本入口 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp script auto-grasp current detection")
    parser.add_argument('--ip', default='localhost', help='Robot IP')
    parser.add_argument('--port', type=int, default=12345, help='Robot port')
    args = parser.parse_args()
    manager = Manager()
    shared_classes = manager.list()
    shared_msgs = manager.list([""])

    # 运行标志
    running = Value(ctypes.c_bool, True)

    def input_thread():
        while running.value:
            inp = input("Enter classes (comma-separated) or 'q': ")
            if inp.strip().lower() == 'q':
                shared_msgs[0] = "Input classes"
                running.value = False
                break
            new = [c.strip() for c in inp.split(',') if c.strip()]
            shared_classes[:] = new if new else []
            shared_msgs[0] = f"Classes: {shared_classes[:]} detected" if new else "Input classes"

    threading.Thread(target=input_thread, daemon=True).start()
    agent = GraspAgent(args.ip, args.port, shared_classes, shared_msgs, running)
    agent.run()
