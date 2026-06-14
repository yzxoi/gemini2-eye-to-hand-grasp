# ---------- 模块导入与全局变量 ----------
import threading
import cv2
import numpy as np
import torch
from PIL import Image
import time
from orbbec_utils import OrbbecCamera
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 共享状态：待检测类别和退出标志
current_classes = []
exit_flag = False
lock = threading.Lock()

# ========== 模型初始化 ==========
model_id = "./grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# ========== 零样本目标检测函数 ==========
def grounding_dino(image: np.ndarray, classes: list) -> list:
    """
    对输入图像执行零样本检测，返回框、标签和分数
    """
    text = "".join(f"{c}." for c in classes)
    image_pil = Image.fromarray(image[:, :, ::-1])  # BGR->RGB
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=0.4, text_threshold=0.4,
        target_sizes=[image_pil.size[::-1]]
    )

# ========== 用户输入线程 ==========
def input_thread():
    """读取命令行输入，更新检测类别或设置退出标志"""
    global current_classes, exit_flag
    while not exit_flag:
        inp = input("请输入检测类别（逗号分隔），或输入 'q' 退出：")
        if inp.strip().lower() == 'q':
            exit_flag = True
            break
        classes = [c.strip() for c in inp.split(',') if c.strip()]
        if classes:
            with lock:
                current_classes = classes
            print(f"已设置检测类别：{current_classes}")
        else:
            print("未输入有效类别，请重新输入。")

# ========== 实时视频检测循环 ==========
def video_loop():
    """启动 Orbbec Gemini 2 流，执行检测并展示结果"""
    # 配置并启动彩色/深度流（深度软件对齐到彩色坐标系）
    cam = OrbbecCamera(color_w=1280, color_h=720, fps=30)

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1280, 720)

    prev_time = time.time()
    try:
        while not exit_flag:
            color_image, depth_u16, depth_m = cam.wait_frames()
            if color_image is None:
                continue

            with lock:
                classes = list(current_classes)

            # 执行检测并绘制结果
            if classes:
                results = grounding_dino(color_image, classes)
                detected = False
                for res in results:
                    for box, label, score in zip(res['boxes'], res['text_labels'], res['scores']):
                        l, t, r, b = map(int, box)
                        cv2.rectangle(color_image, (l, t), (r, b), (0, 255, 0), 2)
                        cv2.putText(color_image, f"{label} {score:.2f}", (l, t - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cx, cy = (l + r) // 2, (t + b) // 2
                        dist = cam.get_distance(depth_m, cx, cy)
                        x_cam, y_cam, z_cam = cam.deproject(cx, cy, dist)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(color_image,
                                    f"X:{x_cam:.3f} Y:{y_cam:.3f} Z:{z_cam:.3f}",
                                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        detected = True
                if not detected:
                    cv2.putText(color_image, "无检测结果", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # 计算并显示FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('Detection', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

# ========== 脚本入口 ==========
if __name__ == '__main__':
    threading.Thread(target=input_thread, daemon=True).start()
    video_loop()
    print("程序已退出。")
