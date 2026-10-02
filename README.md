# 🚗 AI Road Object Detection & Traffic Analytics Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning Computer Vision studio for real-time road object detection, vehicle classification, and traffic analytics. Built with **Ultralytics YOLOv8**, **FastAPI**, **OpenCV**, and an optimized **FFmpeg (H.264)** rendering pipeline.

---

## 🌟 Key Features

- **🖼️ Image Detection Studio:** Upload any road/traffic image from your gallery; tune confidence thresholds (10% - 90%) and get instant bounding box visualizations with per-class object counts.
- **🎥 Long Video Processing with Live Progress:** Process full road videos frame-by-frame. Includes real-time progress bar polling and encodes output with hardware/software `libx264 (yuv420p)` so annotated videos play seamlessly in all modern browsers without codec errors.
- **📹 Live Camera Stream:** High-speed server-side MJPEG video stream directly from your laptop or USB webcam with live FPS, object counts, and inference latency telemetry.
- **🧠 Multi-Model Engine:** Toggle between **YOLOv8 Small (`yolov8s`)** for high detection accuracy and **YOLOv8 Nano (`yolov8n`)** for ultra-fast edge processing.
- **🎯 Intelligent Road Entity Filtering:** Dedicated road filter mode targeting road vehicles (Cars, Trucks, Buses, Motorcycles, Bicycles), pedestrians, and traffic lights while eliminating false positives from non-road classes.

---

## 🛠️ Tech Stack & Architecture

- **Deep Learning / Vision:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), OpenCV (`cv2`)
- **Backend API:** [FastAPI](https://fastapi.tiangolo.com/), Uvicorn (Asynchronous threadpool architecture)
- **Video Transcoding:** FFmpeg (`imageio-ffmpeg`, `libx264`, `yuv420p`)
- **Frontend UI:** Modern Dark-Mode Studio (HTML5, Vanilla CSS3, JavaScript ES6)

---

## 📂 Project Structure

```text
road_object_detection/
├── app.py                   # FastAPI Web Application & API Endpoints
├── detect_image.py          # Standalone CLI Script for Image Detection
├── detect_realtime.py       # Standalone CLI Script for Live Camera
├── train_custom.py          # Script for Custom Model Fine-Tuning
├── dataset.yaml             # COCO/Custom Dataset Configuration
├── requirements.txt         # Project Dependencies
├── .gitignore               # Git Ignore Rules
├── uploads/                 # Temporary storage for uploaded media
└── outputs/                 # Annotated images and rendered H.264 videos
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/syed-jawad123/ai-road-object-detection.git
cd ai-road-object-detection
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Studio
```bash
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 💻 CLI Usage (Without Web Interface)

### Single Image Detection:
```bash
python detect_image.py
```

### Real-Time Live Webcam Detection:
```bash
python detect_realtime.py
```

---

## 🎯 Target Detected Classes (Road Mode)

| Class ID | Object Name | Category |
| :--- | :--- | :--- |
| `0` | Person | Pedestrian |
| `1` | Bicycle | Two-Wheeler |
| `2` | Car | Light Vehicle |
| `3` | Motorcycle | Two-Wheeler |
| `5` | Bus | Heavy Transport |
| `7` | Truck | Heavy Cargo |
| `9` | Traffic Light | Infrastructure |
| `11` | Stop Sign | Infrastructure |

---

## 👤 Author

**Syed Jawad Hussain Shah**
- LinkedIn: [linkedin.com/in/syed-jawad](https://www.linkedin.com)
- GitHub: [@syed-jawad123](https://github.com/syed-jawad123)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
