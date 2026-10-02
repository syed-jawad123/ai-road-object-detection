"""
Road/Object Detection System
Real-time detection using YOLOv8 + OpenCV
Author: Syed Jawad Hussain Shah
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "yolov8n.pt"       # nano (fastest) | yolov8s.pt | yolov8m.pt
VIDEO_SOURCE = 0                 # 0 = webcam | "video.mp4" = video file
CONFIDENCE_THRESHOLD = 0.5      # 50% confidence minimum
TARGET_CLASSES = None            # None = all | [2, 7] = only car & truck

# COCO Class Colors (BGR)
COLORS = {
    0:  (0, 255, 0),     # person = green
    2:  (255, 0, 0),     # car = blue
    3:  (0, 165, 255),   # motorcycle = orange
    5:  (0, 0, 255),     # bus = red
    7:  (128, 0, 128),   # truck = purple
    9:  (0, 255, 255),   # traffic light = yellow
    11: (255, 255, 0),   # stop sign = cyan
}

def load_model(model_path: str) -> YOLO:
    """Load YOLOv8 model"""
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Model loaded successfully!")
    return model

def draw_detections(frame: np.ndarray, results) -> tuple:
    """Draw bounding boxes and labels on frame"""
    total_objects = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            class_id   = int(box.cls[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue
            if TARGET_CLASSES and class_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names[class_id]
            color = COLORS.get(class_id, (255, 255, 255))
            total_objects += 1

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # Label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, total_objects

def add_info_overlay(frame: np.ndarray, fps: float, count: int) -> np.ndarray:
    """Add FPS and detection count HUD"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (250, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}",      (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    cv2.putText(frame, f"Objects: {count}",    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def run_detection():
    """Main detection loop"""
    model = load_model(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source!")
        return

    print("[INFO] Detection running... Press 'Q' to quit | 'S' to save screenshot")
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        results = model(frame, verbose=False)
        frame, total_objects = draw_detections(frame, results)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        frame = add_info_overlay(frame, fps, total_objects)
        cv2.imshow("Road Object Detection — Press Q to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")

if __name__ == "__main__":
    run_detection()
