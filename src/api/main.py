"""
FastAPI inference server for YOLO Vision AI.
Endpoints: /detect (image), /detect/video, /models, /health
"""

from __future__ import annotations

import base64
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.detector import YOLODetector, InferenceResult


app = FastAPI(
    title="YOLO Vision AI API",
    description="Real-time object detection powered by YOLOv8",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model registry — lazy-loaded
_detectors: dict[str, YOLODetector] = {}

AVAILABLE_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


def get_detector(model: str, conf: float, iou: float) -> YOLODetector:
    key = f"{model}_{conf}_{iou}"
    if key not in _detectors:
        _detectors[key] = YOLODetector(model_name=model, conf_threshold=conf, iou_threshold=iou)
    return _detectors[key]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (JPG/PNG)")
    model: str = Field("yolov8n", description="Model variant")
    conf_threshold: float = Field(0.5, ge=0.1, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.1, le=1.0)
    return_annotated: bool = Field(False, description="Return annotated image in response")


class DetectionOut(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]
    center: List[int]


class DetectResponse(BaseModel):
    detections: List[DetectionOut]
    inference_ms: float
    model: str
    image_shape: List[int]
    count: int
    annotated_image_b64: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "available_models": AVAILABLE_MODELS}


@app.get("/models")
def list_models():
    return {"models": AVAILABLE_MODELS}


@app.post("/detect", response_model=DetectResponse)
def detect_image(req: DetectRequest):
    """Detect objects in a base64-encoded image."""
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Model {req.model} not supported. Choose from {AVAILABLE_MODELS}")

    detector = get_detector(req.model, req.conf_threshold, req.iou_threshold)
    result: InferenceResult = detector.detect_from_b64(req.image_b64, annotate=req.return_annotated)

    annotated_b64 = None
    if req.return_annotated and result.annotated_frame is not None:
        _, buf = cv2.imencode(".jpg", result.annotated_frame)
        annotated_b64 = base64.b64encode(buf).decode()

    return DetectResponse(
        detections=[DetectionOut(**d.to_dict()) for d in result.detections],
        inference_ms=result.inference_ms,
        model=result.model_name,
        image_shape=list(result.image_shape),
        count=len(result.detections),
        annotated_image_b64=annotated_b64,
    )


@app.post("/detect/upload")
async def detect_upload(
    file: UploadFile = File(...),
    model: str = "yolov8n",
    conf: float = 0.5,
    annotate: bool = True,
):
    """Upload an image file and get annotated result."""
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode uploaded image")

    detector = get_detector(model, conf, 0.45)
    result = detector.detect_image(frame, annotate=annotate)

    if annotate and result.annotated_frame is not None:
        _, buf = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"X-Detection-Count": str(len(result.detections)),
                                 "X-Inference-MS": str(round(result.inference_ms, 2))})

    return result.to_dict()
