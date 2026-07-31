import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

sys.path.append(str(Path(__file__).parent))

import cv2
from ultralytics import YOLO
import config
from distance_calc import get_final_distance, clear_history

model = YOLO(str(Path(__file__).parent / config.YOLO_WEIGHT))

def draw_warn_box(img, x1, y1, x2, y2, dist, cls_name):
    if dist < config.URGENT_WARN_DIST:
        box_color = (0, 0, 255)
        text_color = (0, 0, 255)
        warn_text = f"{cls_name}: {dist}m  DANGER"
    elif dist < config.NORMAL_WARN_DIST:
        box_color = (0, 255, 255)
        text_color = (0, 255, 255)
        warn_text = f"{cls_name}: {dist}m  WARNING"
    else:
        box_color = (0, 255, 0)
        text_color = (0, 255, 0)
        warn_text = f"{cls_name}: {dist}m Safe"

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
    cv2.putText(img, warn_text, (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return img

def process_frame(img):
    if img is None:
        raise ValueError("输入图像为空，请检查图片路径或视频帧读取")

    results = model(img, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD)
    res_img = img.copy()

    det_count = 0
    if len(results) > 0:
        det_count = len(results[0].boxes)
    print("process_frame: detections =", det_count)

    for box in results[0].boxes:
        cls_idx = int(box.cls)
        cls_name = model.names[cls_idx]
        if cls_name not in ["car", "person"]:
            continue

        xy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(float, xy)

        dist = get_final_distance(cls_name, x1, y1, x2, y2)
        if dist is None:
            continue

        res_img = draw_warn_box(res_img, x1, y1, x2, y2, dist, cls_name)

    return res_img

def run_image(image_path):
    clear_history()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    out_img = process_frame(img)
    cv2.imshow("Image Distance Detection", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result_img.jpg", out_img)

def run_video(video_path):
    clear_history()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (w, h))

    window_name = "Video Distance Warning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("按 空格 暂停/继续，按 q 或 Esc 退出视频播放")

    paused = False
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_out = process_frame(frame)
            out.write(frame_out)
            cv2.imshow(window_name, frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord(' ') or key == ord('p'):
            paused = not paused
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_camera():
    clear_history()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查摄像头是否已连接")
    window_name = "Camera Real-Time Warning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("按 q 或 Esc 退出摄像头")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_out = process_frame(frame)
        cv2.imshow(window_name, frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def choose_file(filetypes, title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=filetypes, title=title)
    root.destroy()
    return path

def choose_image_file():
    return choose_file(
        [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        "请选择图片文件"
    )

def choose_video_file():
    return choose_file(
        [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        "请选择视频文件"
    )

if __name__ == "__main__":
    print("请选择运行模式:")
    print("1: 图片检测")
    print("2: 视频检测")
    print("3: 摄像头实时检测")
    choice = input("输入 1/2/3: ").strip()

    if choice == "1":
        image_path = choose_image_file()
        if not image_path:
            print("未选择图片，程序退出")
        else:
            run_image(image_path)
    elif choice == "2":
        video_path = choose_video_file()
        if not video_path:
            print("未选择视频，程序退出")
        else:
            run_video(video_path)
    elif choice == "3":
        run_camera()
    else:
        print("无效选择，程序退出")


