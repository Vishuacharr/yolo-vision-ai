"""
YOLODetector — wraps Ultralytics YOLOv8 for inference on images, video frames,
webcam streams, and URLs.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "center": list(self.center),
        }


@dataclass
class InferenceResult:
    detections: List[Detection]
    inference_ms: float
    model_name: str
    image_shape: Tuple[int, int]      # (height, width)
    annotated_frame: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "inference_ms": round(self.inference_ms, 2),
            "model": self.model_name,
            "image_shape": list(self.image_shape),
            "count": len(self.detections),
        }


# COCO class color palette (BGR)
_PALETTE = np.random.default_rng(42).integers(0, 255, (80, 3), dtype=np.uint8).tolist()


class YOLODetector:
    """
    High-level YOLOv8 inference wrapper.

    Usage:
        detector = YOLODetector(model_name="yolov8n")
        result = detector.detect_image(frame)
        print(result.to_dict())
    """

    _model_cache: dict = {}

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",         # "cpu" | "cuda" | "auto"
    ):
        self.model_name = model_name
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = self._resolve_device(device)
        self.model = self._load_model(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_image(self, frame: np.ndarray, annotate: bool = True) -> InferenceResult:
        """Run detection on a single BGR frame (numpy array)."""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse_results(results)
        annotated = self._draw_boxes(frame.copy(), detections) if annotate else None

        return InferenceResult(
            detections=detections,
            inference_ms=elapsed_ms,
            model_name=self.model_name,
            image_shape=(h, w),
            annotated_frame=annotated,
        )

    def detect_from_b64(self, image_b64: str, annotate: bool = True) -> InferenceResult:
        """Decode a base64 image string and run detection."""
        img_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image bytes")
        return self.detect_image(frame, annotate=annotate)

    def detect_from_file(self, path: Union[str, Path], annotate: bool = True) -> InferenceResult:
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.detect_image(frame, annotate=annotate)

    def stream_webcam(self, source: int = 0, show: bool = True) -> None:
        """Real-time webcam inference loop."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam source: {source}")

        print(f"[YOLODetector] Streaming from source {source}. Press 'q' to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.detect_image(frame, annotate=True)
                if show and result.annotated_frame is not None:
                    cv2.imshow(f"YOLO Vision AI — {self.model_name}", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        names = self.model.names

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        class_name=names[cls_id],
                        class_id=cls_id,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections

    @staticmethod
    def _draw_boxes(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _PALETTE[det.class_id % len(_PALETTE)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    @classmethod
    def _load_model(cls, model_name: str) -> YOLO:
        if model_name not in cls._model_cache:
            cls._model_cache[model_name] = YOLO(f"{model_name}.pt")
        return cls._model_cache[model_name]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

# Alias for backwards compatibility
DetectionResult = InferenceResult
