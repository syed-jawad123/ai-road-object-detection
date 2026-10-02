"""
Train YOLOv8 on Custom Dataset
"""

from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data    = "dataset.yaml",
        epochs  = 50,
        imgsz   = 640,
        batch   = 16,
        name    = "road_detector",
        device  = "cpu",   # Use "0" if you have a GPU
        patience= 10,
        save    = True,
    )

    print("\n[DONE] Training complete!")
    print("Best model: runs/detect/road_detector/weights/best.pt")

    metrics = model.val()
    print(f"\n[METRICS]")
    print(f"  mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"  mAP@0.5-0.95: {metrics.box.map:.3f}")
    print(f"  Precision:    {metrics.box.mp:.3f}")
    print(f"  Recall:       {metrics.box.mr:.3f}")

if __name__ == "__main__":
    train_model()
