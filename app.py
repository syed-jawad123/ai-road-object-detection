"""
AI Road & Object Detection Studio
FastAPI + YOLOv8 + Multi-Model Engine (Nano / Small) + Road Objects Filtering
Supports Gallery Images, Long Videos with Progress Bar, and Real-Time Webcam
Author: Syed Jawad Hussain Shah
"""

import os
import io
import time
import base64
import cv2
import numpy as np
import subprocess
import imageio_ffmpeg
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="AI Road Object Detection Studio")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pre-load Models in Memory
MODELS = {
    "yolov8s": YOLO(os.path.join(BASE_DIR, "yolov8s.pt")),
    "yolov8n": YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
}

# COCO Road Vehicle & Pedestrian Class IDs
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 11: stop sign
ROAD_CLASS_IDS = [0, 1, 2, 3, 5, 7, 9, 11]

def get_model(name: str):
    if name not in MODELS:
        MODELS[name] = YOLO(os.path.join(BASE_DIR, f"{name}.pt"))
    return MODELS[name]

# Global State for Video Progress
video_progress = {"percent": 0, "current": 0, "total": 0, "status": "idle"}

# Global State for Live Camera
camera_running = False
camera_cap = None
camera_conf = 0.25
camera_model_name = "yolov8s"
camera_filter = "road"
camera_stats = {"fps": 0, "objects": 0, "latency": 0, "counts": {}}

def generate_live_frames():
    global camera_running, camera_cap, camera_conf, camera_model_name, camera_filter, camera_stats
    camera_cap = cv2.VideoCapture(0)
    prev_time = time.time()
    
    current_model = get_model(camera_model_name)
    target_classes = ROAD_CLASS_IDS if camera_filter == "road" else None
    
    while camera_running and camera_cap.isOpened():
        ret, frame = camera_cap.read()
        if not ret:
            break
            
        t0 = time.time()
        res = current_model.predict(source=frame, conf=camera_conf, classes=target_classes, imgsz=480, verbose=False)
        inf_ms = round((time.time() - t0) * 1000, 1)
        
        annotated = res[0].plot(line_width=3)
        
        curr_time = time.time()
        fps = round(1.0 / (curr_time - prev_time + 1e-6), 1)
        prev_time = curr_time
        
        counts = {}
        for box in res[0].boxes:
            cls_id = int(box.cls[0])
            name = res[0].names.get(cls_id, f"Class {cls_id}").capitalize()
            counts[name] = counts.get(name, 0) + 1
            
        camera_stats = {
            "fps": fps,
            "objects": len(res[0].boxes),
            "latency": inf_ms,
            "counts": counts
        }
        
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    if camera_cap:
        camera_cap.release()
        camera_cap = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Road & Object Detection Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: #1a2234;
            --border-color: #2a3449;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-purple: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        
        header { text-align: center; margin-bottom: 24px; }
        header h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #38bdf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
        header p { color: var(--text-muted); font-size: 1rem; }

        /* Global Settings Bar */
        .settings-bar { display: flex; justify-content: center; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .setting-chip { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 9999px; padding: 8px 20px; display: flex; align-items: center; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .setting-chip label { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
        .setting-chip select { background: transparent; color: var(--accent-cyan); border: none; font-weight: 600; font-size: 0.92rem; cursor: pointer; outline: none; }
        .setting-chip select option { background: #121826; color: #f8fafc; }

        .tabs { display: flex; justify-content: center; gap: 12px; margin-bottom: 24px; }
        .tab-btn { background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-muted); padding: 12px 28px; border-radius: 9999px; cursor: pointer; font-weight: 600; font-size: 1rem; transition: all 0.2s; }
        .tab-btn:hover { border-color: var(--accent-blue); color: var(--text-main); }
        .tab-btn.active { background: var(--accent-blue); color: white; border-color: var(--accent-blue); box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4); }

        .panel { display: none; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 16px; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
        .panel.active { display: block; }

        .controls { display: grid; grid-template-columns: 1fr auto; gap: 20px; align-items: center; margin-bottom: 24px; background: var(--bg-card); padding: 16px 20px; border-radius: 12px; border: 1px solid var(--border-color); }
        .slider-group { display: flex; align-items: center; gap: 12px; }
        .slider-group label { font-size: 0.9rem; color: var(--text-muted); }
        .slider-group input { width: 140px; accent-color: var(--accent-blue); }
        .slider-value { font-weight: 600; color: var(--accent-cyan); min-width: 40px; }

        .upload-dropzone { border: 2px dashed var(--border-color); border-radius: 14px; padding: 40px 20px; text-align: center; cursor: pointer; background: rgba(26, 34, 52, 0.5); transition: all 0.2s; }
        .upload-dropzone:hover { border-color: var(--accent-blue); background: rgba(59, 130, 246, 0.05); }
        .upload-dropzone input { display: none; }
        .upload-icon { font-size: 40px; margin-bottom: 12px; }

        .btn { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; padding: 12px 26px; border-radius: 10px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-size: 0.95rem; }
        .btn:hover { opacity: 0.95; transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

        .results-container { margin-top: 28px; display: none; }
        .preview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        @media (max-width: 768px) { .preview-grid { grid-template-columns: 1fr; } }
        .preview-card { background: var(--bg-card); border-radius: 12px; padding: 14px; border: 1px solid var(--border-color); }
        .preview-card h3 { font-size: 0.95rem; color: var(--text-muted); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        .preview-card img, .preview-card video { width: 100%; border-radius: 8px; display: block; max-height: 480px; object-fit: contain; background: #000; }

        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 20px; }
        .stat-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px 18px; }
        .stat-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent-cyan); margin-top: 4px; }

        .objects-list { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 18px; }
        .objects-list h4 { margin-bottom: 12px; font-size: 1rem; color: var(--text-main); }
        .tag-cloud { display: flex; flex-wrap: wrap; gap: 8px; }
        .tag { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); color: #93c5fd; padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; font-weight: 500; }
        .tag-count { background: #3b82f6; color: white; padding: 2px 7px; border-radius: 9999px; font-size: 0.75rem; margin-left: 6px; }

        /* Progress Bar */
        .progress-box { display: none; margin: 24px 0; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
        .progress-track { width: 100%; height: 12px; background: rgba(255,255,255,0.08); border-radius: 9999px; overflow: hidden; margin-top: 10px; }
        .progress-fill { width: 0%; height: 100%; background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981); border-radius: 9999px; transition: width 0.3s ease; }
        
        .loading-spinner { display: none; text-align: center; margin: 30px 0; }
        .spinner { width: 44px; height: 44px; border: 4px solid rgba(255,255,255,0.1); border-top-color: var(--accent-blue); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 12px; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚗 AI Road Object Detection Studio</h1>
            <p>High-Accuracy Deep Learning Vision with YOLOv8 | Image, Video & Live Webcam Modes</p>
        </header>

        <!-- GLOBAL SETTINGS BAR -->
        <div class="settings-bar">
            <div class="setting-chip">
                <label>🧠 AI Model Engine:</label>
                <select id="global-model">
                    <option value="yolov8s" selected>🎯 YOLOv8 Small (High Accuracy - Recommended)</option>
                    <option value="yolov8n">⚡ YOLOv8 Nano (Super Fast)</option>
                </select>
            </div>
            <div class="setting-chip">
                <label>🎯 Detection Filter:</label>
                <select id="global-filter">
                    <option value="road" selected>🚗 Road Vehicles & Pedestrians Only (No Trains/Boats)</option>
                    <option value="all">🌐 All 80 COCO Objects (Detect Everything)</option>
                </select>
            </div>
        </div>

        <div class="tabs">
            <button class="tab-btn active" id="btn-tab-image" onclick="switchTab('image-panel', this)">🖼️ Image Upload</button>
            <button class="tab-btn" id="btn-tab-video" onclick="switchTab('video-panel', this)">🎥 Video Upload</button>
            <button class="tab-btn" id="btn-tab-live" onclick="switchTab('live-panel', this)">📹 Live Camera</button>
        </div>

        <!-- ==================== 1. IMAGE PANEL ==================== -->
        <div id="image-panel" class="panel active">
            <div class="controls">
                <div class="slider-group">
                    <label>Detection Confidence:</label>
                    <input type="range" id="img-conf" min="0.1" max="0.9" step="0.05" value="0.25" oninput="updateConf(this, 'img-conf-val')">
                    <span id="img-conf-val" class="slider-value">25%</span>
                </div>
                <button class="btn" id="img-detect-btn" onclick="detectImage()" disabled>Run Detection ⚡</button>
            </div>

            <div class="upload-dropzone" onclick="document.getElementById('img-file-input').click()">
                <div class="upload-icon">📸</div>
                <h3 id="img-filename">Click or Drag & Drop Image Here</h3>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px;">Supports JPG, PNG, WEBP, JPEG</p>
                <input type="file" id="img-file-input" accept="image/*" onchange="handleImageSelected(this)">
            </div>

            <div id="img-loading" class="loading-spinner">
                <div class="spinner"></div>
                <p>Analyzing image with YOLOv8...</p>
            </div>

            <div id="img-results" class="results-container">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Objects</div>
                        <div class="stat-value" id="stat-total-objs">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Inference Time</div>
                        <div class="stat-value" id="stat-inf-time">0 ms</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Active Model</div>
                        <div class="stat-value" id="stat-active-model" style="font-size: 1.2rem; color: #a78bfa;">YOLOv8 Small</div>
                    </div>
                </div>

                <div class="preview-grid">
                    <div class="preview-card">
                        <h3>Original Image</h3>
                        <img id="img-orig-preview" src="">
                    </div>
                    <div class="preview-card">
                        <h3>YOLOv8 Detection (With Bounding Boxes)</h3>
                        <img id="img-detected-preview" src="">
                    </div>
                </div>

                <div class="objects-list">
                    <h4>Detected Categories & Counts:</h4>
                    <div class="tag-cloud" id="img-tag-cloud"></div>
                </div>
            </div>
        </div>

        <!-- ==================== 2. VIDEO PANEL ==================== -->
        <div id="video-panel" class="panel">
            <div class="controls">
                <div class="slider-group">
                    <label>Detection Confidence:</label>
                    <input type="range" id="vid-conf" min="0.1" max="0.9" step="0.05" value="0.25" oninput="updateConf(this, 'vid-conf-val')">
                    <span id="vid-conf-val" class="slider-value">25%</span>
                </div>
                <button class="btn" id="vid-detect-btn" onclick="detectVideo()" disabled>Process Video ⚡</button>
            </div>

            <div class="upload-dropzone" onclick="document.getElementById('vid-file-input').click()">
                <div class="upload-icon">🎬</div>
                <h3 id="vid-filename">Click or Drag & Drop Road Video Here</h3>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px;">Supports MP4, AVI, MOV (full video will be annotated)</p>
                <input type="file" id="vid-file-input" accept="video/*" onchange="handleVideoSelected(this)">
            </div>

            <!-- Live Progress Bar -->
            <div id="vid-progress-container" class="progress-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span id="vid-progress-label" style="font-weight: 600; color: var(--text-main);">Processing Video Frames...</span>
                    <span id="vid-progress-pct" style="font-weight: 700; color: var(--accent-cyan);">0%</span>
                </div>
                <div class="progress-track">
                    <div id="vid-progress-bar" class="progress-fill"></div>
                </div>
                <p id="vid-progress-details" style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">Encoding frames with H.264...</p>
            </div>

            <div id="vid-results" class="results-container">
                <div class="preview-card" style="margin-bottom: 20px;">
                    <h3>Annotated Video Output (H.264 High Definition)</h3>
                    <video id="vid-output-player" controls style="width: 100%; border-radius: 8px;"></video>
                    <div style="margin-top: 14px; text-align: right;">
                        <a id="vid-download-btn" href="#" download="yolo_detected_road_video.mp4" class="btn" style="display: inline-block; text-decoration: none;">⬇️ Download Full Video</a>
                    </div>
                </div>
                <div class="objects-list">
                    <h4>Video Processing Summary:</h4>
                    <div class="tag-cloud" id="vid-summary-tags"></div>
                </div>
            </div>
        </div>

        <!-- ==================== 3. LIVE WEBCAM PANEL ==================== -->
        <div id="live-panel" class="panel">
            <div class="controls">
                <div class="slider-group">
                    <label>Detection Confidence:</label>
                    <input type="range" id="cam-conf" min="0.1" max="0.9" step="0.05" value="0.25" oninput="updateConf(this, 'cam-conf-val')">
                    <span id="cam-conf-val" class="slider-value">25%</span>
                </div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="btn" id="cam-start-btn" onclick="startLiveCamera()" style="background: linear-gradient(135deg, #10b981, #059669);">▶️ Start Camera</button>
                    <button class="btn" id="cam-stop-btn" onclick="stopLiveCamera()" disabled style="background: linear-gradient(135deg, #ef4444, #dc2626);">⏹️ Stop Camera</button>
                    <button class="btn" id="cam-snap-btn" onclick="snapshotLiveCamera()" disabled style="background: linear-gradient(135deg, #8b5cf6, #7c3aed);">📸 Snapshot</button>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Camera Status</div>
                    <div class="stat-value" id="cam-status-msg" style="font-size: 1.1rem; color: #94a3b8;">Inactive ⚪</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Live Speed</div>
                    <div class="stat-value" id="cam-fps" style="color: var(--accent-green);">0 FPS</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Detected Objects</div>
                    <div class="stat-value" id="cam-objects">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Inference Latency</div>
                    <div class="stat-value" id="cam-latency" style="font-size: 1.2rem; color: #a78bfa;">0 ms</div>
                </div>
            </div>

            <div class="preview-card" style="margin-top: 20px; text-align: center;">
                <h3>Live AI Detection Stream</h3>
                <div style="position: relative; max-width: 900px; margin: 0 auto; overflow: hidden; border-radius: 10px; background: #000; min-height: 420px; display: flex; align-items: center; justify-content: center;">
                    <div id="cam-placeholder" style="color: var(--text-muted); font-size: 1rem;">
                        Click <b>"▶️ Start Camera"</b> to stream live AI object detection.
                    </div>
                    <img id="live-stream-img" src="" style="width: 100%; border-radius: 10px; display: none; object-fit: contain;">
                </div>
            </div>

            <div class="objects-list" style="margin-top: 20px;">
                <h4>Real-time Detected Categories:</h4>
                <div class="tag-cloud" id="cam-tag-cloud">
                    <span style="color: var(--text-muted); font-size: 0.9rem;">Start the camera to begin live detection.</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedImage = null;
        let selectedVideo = null;
        let isCameraActive = false;
        let camStatsInterval = null;

        function switchTab(panelId, btn) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(panelId).classList.add('active');
            btn.classList.add('active');
        }

        function updateConf(slider, labelId) {
            document.getElementById(labelId).innerText = Math.round(slider.value * 100) + '%';
        }

        // ==================== 1. IMAGE HANDLING ====================
        function handleImageSelected(input) {
            if (input.files && input.files[0]) {
                selectedImage = input.files[0];
                document.getElementById('img-filename').innerText = "Selected: " + selectedImage.name;
                document.getElementById('img-detect-btn').disabled = false;
                
                const reader = new FileReader();
                reader.onload = e => {
                    document.getElementById('img-orig-preview').src = e.target.result;
                };
                reader.readAsDataURL(selectedImage);
            }
        }

        async function detectImage() {
            if (!selectedImage) return;
            const btn = document.getElementById('img-detect-btn');
            const loading = document.getElementById('img-loading');
            const results = document.getElementById('img-results');

            btn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';

            const formData = new FormData();
            formData.append('file', selectedImage);
            formData.append('confidence', document.getElementById('img-conf').value);
            formData.append('model_name', document.getElementById('global-model').value);
            formData.append('target_filter', document.getElementById('global-filter').value);

            try {
                const res = await fetch('/predict/image', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.success) {
                    document.getElementById('img-detected-preview').src = data.annotated_image;
                    document.getElementById('stat-total-objs').innerText = data.total_objects;
                    document.getElementById('stat-inf-time').innerText = data.inference_ms + ' ms';
                    document.getElementById('stat-active-model').innerText = data.model_used;

                    const tagCloud = document.getElementById('img-tag-cloud');
                    tagCloud.innerHTML = '';
                    if (Object.keys(data.counts).length === 0) {
                        tagCloud.innerHTML = '<span style="color: var(--text-muted)">No objects detected at this confidence level.</span>';
                    } else {
                        for (const [name, count] of Object.entries(data.counts)) {
                            tagCloud.innerHTML += `<div class="tag">${name} <span class="tag-count">${count}</span></div>`;
                        }
                    }
                    results.style.display = 'block';
                } else {
                    alert('Detection failed: ' + data.error);
                }
            } catch (err) {
                alert('Server error: ' + err.message);
            } finally {
                loading.style.display = 'none';
                btn.disabled = false;
            }
        }

        // ==================== 2. VIDEO HANDLING WITH LIVE PROGRESS ====================
        function handleVideoSelected(input) {
            if (input.files && input.files[0]) {
                selectedVideo = input.files[0];
                document.getElementById('vid-filename').innerText = "Selected: " + selectedVideo.name;
                document.getElementById('vid-detect-btn').disabled = false;
            }
        }

        async function detectVideo() {
            if (!selectedVideo) return;
            const btn = document.getElementById('vid-detect-btn');
            const progressBox = document.getElementById('vid-progress-container');
            const progressBar = document.getElementById('vid-progress-bar');
            const progressPct = document.getElementById('vid-progress-pct');
            const progressDetails = document.getElementById('vid-progress-details');
            const results = document.getElementById('vid-results');

            btn.disabled = true;
            progressBox.style.display = 'block';
            results.style.display = 'none';
            progressBar.style.width = '0%';
            progressPct.innerText = '0%';
            progressDetails.innerText = 'Uploading video and starting YOLOv8 engine...';

            const pollInterval = setInterval(async () => {
                try {
                    const r = await fetch('/predict/video/progress');
                    const p = await r.json();
                    if (p.status === 'processing') {
                        progressBar.style.width = p.percent + '%';
                        progressPct.innerText = p.percent + '%';
                        progressDetails.innerText = `Processed ${p.current} of ${p.total} frames...`;
                    }
                } catch(e) {}
            }, 600);

            const formData = new FormData();
            formData.append('file', selectedVideo);
            formData.append('confidence', document.getElementById('vid-conf').value);
            formData.append('model_name', document.getElementById('global-model').value);
            formData.append('target_filter', document.getElementById('global-filter').value);

            try {
                const res = await fetch('/predict/video', { method: 'POST', body: formData });
                const data = await res.json();
                clearInterval(pollInterval);

                if (data.success) {
                    progressBar.style.width = '100%';
                    progressPct.innerText = '100%';
                    progressDetails.innerText = `Completed in ${data.processing_sec}s!`;

                    const player = document.getElementById('vid-output-player');
                    player.src = data.video_url;
                    player.load();
                    document.getElementById('vid-download-btn').href = data.video_url;

                    const summaryTags = document.getElementById('vid-summary-tags');
                    summaryTags.innerHTML = `
                        <div class="tag">Total Detections <span class="tag-count">${data.total_detections || 0}</span></div>
                        <div class="tag">Processed Frames <span class="tag-count">${data.frames_processed}</span></div>
                        <div class="tag">FPS <span class="tag-count">${data.fps}</span></div>
                        <div class="tag">Processing Time <span class="tag-count">${data.processing_sec}s</span></div>
                    `;
                    results.style.display = 'block';
                } else {
                    alert('Video processing failed: ' + data.error);
                }
            } catch (err) {
                clearInterval(pollInterval);
                alert('Server error: ' + err.message);
            } finally {
                progressBox.style.display = 'none';
                btn.disabled = false;
            }
        }

        // ==================== 3. LIVE WEBCAM ENGINE (HIGH SPEED) ====================
        async function startLiveCamera() {
            const startBtn = document.getElementById('cam-start-btn');
            const stopBtn = document.getElementById('cam-stop-btn');
            const snapBtn = document.getElementById('cam-snap-btn');
            const statusMsg = document.getElementById('cam-status-msg');
            const streamImg = document.getElementById('live-stream-img');
            const placeholder = document.getElementById('cam-placeholder');
            const conf = document.getElementById('cam-conf').value || 0.25;
            const modelName = document.getElementById('global-model').value;
            const filterMode = document.getElementById('global-filter').value;

            try {
                const res = await fetch(`/start_camera?confidence=${conf}&model_name=${modelName}&target_filter=${filterMode}`);
                const data = await res.json();

                if (data.success) {
                    isCameraActive = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    snapBtn.disabled = false;
                    statusMsg.innerText = "Live Active 🟢";
                    statusMsg.style.color = "var(--accent-green)";

                    placeholder.style.display = "none";
                    streamImg.style.display = "block";
                    streamImg.src = "/video_feed?" + Date.now();

                    camStatsInterval = setInterval(async () => {
                        try {
                            const r = await fetch('/camera_status');
                            const s = await r.json();
                            document.getElementById('cam-fps').innerText = s.fps + " FPS";
                            document.getElementById('cam-objects').innerText = s.objects;
                            document.getElementById('cam-latency').innerText = s.latency + " ms";

                            const tagCloud = document.getElementById('cam-tag-cloud');
                            tagCloud.innerHTML = '';
                            if (Object.keys(s.counts).length === 0) {
                                tagCloud.innerHTML = '<span style="color: var(--text-muted)">No objects detected in current frame.</span>';
                            } else {
                                for (const [name, count] of Object.entries(s.counts)) {
                                    tagCloud.innerHTML += `<div class="tag">${name} <span class="tag-count">${count}</span></div>`;
                                }
                            }
                        } catch(e) {}
                    }, 800);
                }
            } catch (err) {
                alert("Failed to start camera: " + err.message);
            }
        }

        async function stopLiveCamera() {
            isCameraActive = false;
            if (camStatsInterval) clearInterval(camStatsInterval);

            const startBtn = document.getElementById('cam-start-btn');
            const stopBtn = document.getElementById('cam-stop-btn');
            const snapBtn = document.getElementById('cam-snap-btn');
            const statusMsg = document.getElementById('cam-status-msg');
            const streamImg = document.getElementById('live-stream-img');
            const placeholder = document.getElementById('cam-placeholder');

            streamImg.src = "";
            streamImg.style.display = "none";
            placeholder.style.display = "block";

            await fetch('/stop_camera');

            startBtn.disabled = false;
            stopBtn.disabled = true;
            snapBtn.disabled = true;
            statusMsg.innerText = "Stopped 🔴";
            statusMsg.style.color = "var(--text-muted)";
            document.getElementById('cam-fps').innerText = "0 FPS";
        }

        function snapshotLiveCamera() {
            const streamImg = document.getElementById('live-stream-img');
            const canvas = document.createElement('canvas');
            canvas.width = streamImg.naturalWidth || 640;
            canvas.height = streamImg.naturalHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(streamImg, 0, 0, canvas.width, canvas.height);
            
            const link = document.createElement('a');
            link.download = `yolo_live_snapshot_${Date.now()}.jpg`;
            link.href = canvas.toDataURL('image/jpeg', 0.95);
            link.click();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/predict/image")
def predict_image(
    file: UploadFile = File(...), 
    confidence: float = Form(0.25),
    model_name: str = Form("yolov8s"),
    target_filter: str = Form("road")
):
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        current_model = get_model(model_name)
        target_classes = ROAD_CLASS_IDS if target_filter == "road" else None

        t0 = time.time()
        results = current_model.predict(source=img, conf=confidence, classes=target_classes, imgsz=640, verbose=False)
        inf_time = round((time.time() - t0) * 1000, 1)

        result = results[0]
        annotated_bgr = result.plot(line_width=3)
        
        _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        counts = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = result.names.get(cls_id, f"Class {cls_id}").capitalize()
            counts[name] = counts.get(name, 0) + 1

        label_model = "YOLOv8 Small (Accurate)" if "s" in model_name else "YOLOv8 Nano (Fast)"
        return {
            "success": True,
            "total_objects": len(result.boxes),
            "counts": counts,
            "inference_ms": inf_time,
            "model_used": label_model,
            "annotated_image": data_uri
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/predict/video/progress")
def get_video_progress():
    global video_progress
    return video_progress

@app.post("/predict/video")
def predict_video(
    file: UploadFile = File(...), 
    confidence: float = Form(0.25),
    model_name: str = Form("yolov8s"),
    target_filter: str = Form("road")
):
    global video_progress
    try:
        video_progress = {"percent": 0, "current": 0, "total": 0, "status": "processing"}
        
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        output_filename = f"detected_{int(time.time())}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(input_path, "wb") as f:
            f.write(file.file.read())

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_progress["total"] = total_frames

        current_model = get_model(model_name)
        target_classes = ROAD_CLASS_IDS if target_filter == "road" else None

        # High-Speed FFmpeg Pipe (libx264, yuv420p)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-crf', '24',
            output_path
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        frames_count = 0
        total_detections = 0
        t0 = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            res = current_model.predict(source=frame, conf=confidence, classes=target_classes, imgsz=480, verbose=False)
            annotated = res[0].plot(line_width=3)
            proc.stdin.write(annotated.tobytes())
            frames_count += 1
            total_detections += len(res[0].boxes)
            
            if frames_count % 10 == 0 or frames_count == total_frames:
                video_progress["current"] = frames_count
                video_progress["percent"] = min(99, int((frames_count / total_frames) * 100))

        proc.stdin.close()
        proc.wait()
        cap.release()
        total_time = round(time.time() - t0, 1)
        video_progress = {"percent": 100, "current": frames_count, "total": total_frames, "status": "completed"}

        return {
            "success": True,
            "video_url": f"/outputs/{output_filename}",
            "frames_processed": frames_count,
            "total_detections": total_detections,
            "fps": fps,
            "processing_sec": total_time
        }
    except Exception as e:
        video_progress = {"percent": 0, "current": 0, "total": 0, "status": "error"}
        return {"success": False, "error": str(e)}

@app.get("/start_camera")
def start_camera_api(confidence: float = 0.25, model_name: str = "yolov8s", target_filter: str = "road"):
    global camera_running, camera_conf, camera_model_name, camera_filter
    camera_conf = confidence
    camera_model_name = model_name
    camera_filter = target_filter
    camera_running = True
    return {"success": True, "message": "Camera started"}

@app.get("/stop_camera")
def stop_camera_api():
    global camera_running, camera_cap
    camera_running = False
    if camera_cap:
        camera_cap.release()
        camera_cap = None
    return {"success": True, "message": "Camera stopped"}

@app.get("/camera_status")
def camera_status_api():
    global camera_stats
    return camera_stats

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_live_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/outputs/{filename}")
def get_output_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
