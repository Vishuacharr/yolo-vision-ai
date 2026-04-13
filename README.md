# 👁️ YOLO Vision AI

> **Real-time object detection and tracking** using YOLOv8 with a FastAPI inference server, Streamlit web UI, webcam/video/image support, and Docker deployment.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv)](https://opencv.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Demo

```
Input: webcam / video file / image URL
  ↓
YOLOv8 Inference (GPU/CPU)
  ↓
Bounding boxes + class labels + confidence scores
  ↓
REST API response | Streamlit overlay | RTSP stream
```

**Supported models:** `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
**COCO classes:** 80 objects (person, car, dog, laptop, ...)
**Custom training:** plug-and-play fine-tuning pipeline included

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🚀 Real-time inference | 30+ FPS on GPU, 10+ FPS on CPU |
| 📷 Multi-source input | Webcam, video file, image, URL, RTSP stream |
| 🌐 REST API | FastAPI with `/detect` image endpoint |
| 🖼️ Streamlit UI | Live annotated video feed in browser |
| 🎯 Object tracking | DeepSORT multi-object tracking |
| 🏋️ Custom training | Fine-tune on your own dataset |
| 🐳 Docker | One-command deployment |
| 📊 Metrics dashboard | Precision, recall, mAP per class |

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)
```bash
git clone https://github.com/Vishuacharr/yolo-vision-ai
cd yolo-vision-ai
docker-compose up
# Open http://localhost:8501 for Streamlit UI
# Open http://localhost:8000/docs for API docs
```

### Option 2 — Local
```bash
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload --port 8000

# Run Streamlit UI (separate terminal)
streamlit run src/ui/app.py

# Run webcam detection
python src/detect.py --source 0 --model yolov8n

# Run on video file
python src/detect.py --source video.mp4 --model yolov8m --save
```

---

## 📡 API Usage

```python
import requests, base64

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/detect", json={
    "image_b64": img_b64,
    "model": "yolov8n",
    "conf_threshold": 0.5,
    "iou_threshold": 0.45
})

print(response.json())
# {
#   "detections": [
#     {"class": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2]},
#     {"class": "car",    "confidence": 0.88, "bbox": [x1, y1, x2, y2]}
#   ],
#   "inference_ms": 23.4,
#   "model": "yolov8n"
# }
```

---

## 📁 Project Structure

```
yolo-vision-ai/
├── src/
│   ├── detect.py           # CLI entrypoint
│   ├── detector.py         # YOLODetector class
│   ├── tracker.py          # DeepSORT object tracking
│   ├── api/
│   │   ├── main.py         # FastAPI app
│   │   └── schemas.py      # Request/response models
│   ├── ui/
│   │   └── app.py          # Streamlit interface
│   └── training/
│       ├── train.py        # Custom dataset fine-tuning
│       └── dataset.py      # YOLO dataset preparation
├── models/                 # Downloaded YOLO weights
├── tests/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🏋️ Custom Training

```bash
# Prepare dataset in YOLO format
python src/training/dataset.py --data ./my_dataset --split 0.8

# Train
python src/training/train.py \
  --model yolov8s \
  --data dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16

# Results saved to runs/train/
```

---

## 🧠 Tech Stack

- **Detection**: Ultralytics YOLOv8
- **Tracking**: DeepSORT
- **API**: FastAPI + uvicorn
- **UI**: Streamlit + OpenCV
- **Deployment**: Docker + NVIDIA Container Toolkit (GPU)

## 📄 License

MIT — see [LICENSE](LICENSE)
