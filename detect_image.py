"""
Single Image Object Detection
Usage: python detect_image.py road_image.jpg
"""

from ultralytics import YOLO
import sys

def detect_image(image_path: str):
    model   = YOLO("yolov8n.pt")
    results = model(image_path)

    for result in results:
        print(f"\n[DETECTIONS in '{image_path}']")
        print(f"Total objects found: {len(result.boxes)}\n")

        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            coords     = [round(c, 1) for c in box.xyxy[0].tolist()]
            print(f"  • {class_name:15s} {confidence:.2%} confidence | Box: {coords}")

        result.save(filename="result_detected.jpg")
        print("\n[INFO] Result saved as: result_detected.jpg")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    detect_image(path)
