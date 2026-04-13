"""
YOLODetector unit tests — runs without GPU, ultralytics, or cv2 installed.
All heavy dependencies are mocked at module level before any src import.
"""
import sys
import types
import numpy as np

# ---------------------------------------------------------------------------
# Mock ultralytics + cv2 before importing src
# ---------------------------------------------------------------------------

class _FakeBox:
    def __init__(self, x1, y1, x2, y2, cls_id, conf):
        self.xyxy = [np.array([x1, y1, x2, y2], dtype=float)]
        self.cls  = [np.array(cls_id, dtype=float)]
        self.conf = [np.array(conf, dtype=float)]

class _FakeResult:
    orig_shape = (480, 640)
    boxes = None   # override per test

class _FakeYOLO:
    names = {0: "person", 2: "car", 5: "bus"}
    def __init__(self, path): self.path = path
    def predict(self, source, **kw): return [_FakeResult()]
    def to(self, device): return self

_ultra = types.ModuleType("ultralytics")
_ultra.YOLO = _FakeYOLO
sys.modules.setdefault("ultralytics", _ultra)

# Mock cv2 — annotate=False skips drawing, but cv2 must be importable
_cv2 = types.ModuleType("cv2")
_cv2.imread = lambda path, *a: np.zeros((480, 640, 3), dtype="uint8")
_cv2.rectangle = lambda *a, **kw: None
_cv2.putText = lambda *a, **kw: None
_cv2.getTextSize = lambda *a, **kw: ((60, 15), 0)
_cv2.FONT_HERSHEY_SIMPLEX = 0
_cv2.VideoCapture = type("VC", (), {
    "read": lambda s: (True, np.zeros((480, 640, 3), "uint8")),
    "release": lambda s: None,
})
sys.modules.setdefault("cv2", _cv2)

# ---------------------------------------------------------------------------
import pytest
from src.detector import Detection, InferenceResult, YOLODetector


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

def test_detection_center_calculation():
    d = Detection(class_name="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 50))
    assert d.center == (50, 25)


def test_detection_center_off_origin():
    d = Detection(class_name="person", class_id=0, confidence=0.8, bbox=(10, 20, 50, 80))
    assert d.center == (30, 50)


def test_detection_to_dict_has_expected_keys():
    d = Detection(class_name="person", class_id=0, confidence=0.8, bbox=(10, 20, 30, 40))
    dd = d.to_dict()
    for key in ("class", "class_id", "confidence", "bbox", "center"):
        assert key in dd, f"Missing key: {key}"


def test_detection_confidence_in_dict():
    d = Detection(class_name="bus", class_id=5, confidence=0.75432, bbox=(0, 0, 10, 10))
    assert d.to_dict()["confidence"] == round(0.75432, 4)


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------

def test_inference_result_count_in_dict():
    r = InferenceResult(
        detections=[Detection("car", 2, 0.9, (0, 0, 100, 100))],
        inference_ms=15.0,
        model_name="yolov8n",
        image_shape=(480, 640),
    )
    d = r.to_dict()
    assert d["count"] == 1
    assert d["model"] == "yolov8n"
    assert d["inference_ms"] == 15.0


def test_inference_result_empty():
    r = InferenceResult([], 10.0, "yolov8n", (480, 640))
    assert r.to_dict()["count"] == 0


def test_inference_result_image_shape_serialized():
    r = InferenceResult([], 5.0, "yolov8s", (720, 1280))
    assert r.to_dict()["image_shape"] == [720, 1280]


# ---------------------------------------------------------------------------
# YOLODetector integration (uses mocked YOLO)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_model_cache():
    """Reset class-level model cache between tests."""
    YOLODetector._model_cache.clear()
    yield
    YOLODetector._model_cache.clear()


def test_detector_creates_model_on_init():
    det = YOLODetector(model_name="yolov8n", device="cpu")
    assert det.model is not None
    assert det.model_name == "yolov8n"


def test_detect_image_no_boxes():
    det = YOLODetector(model_name="yolov8n", device="cpu")
    img = np.zeros((480, 640, 3), dtype="uint8")
    result = det.detect_image(img, annotate=False)
    assert isinstance(result, InferenceResult)
    assert result.image_shape == (480, 640)
    assert result.model_name == "yolov8n"
    assert len(result.detections) == 0


def test_detect_image_with_boxes():
    """Inject boxes into the fake YOLO result to verify parsing."""
    class _ResultWithBoxes(_FakeResult):
        boxes = [
            _FakeBox(10, 20, 100, 200, 0, 0.92),
            _FakeBox(300, 150, 580, 400, 2, 0.78),
        ]

    class _YOLOWithBoxes(_FakeYOLO):
        def predict(self, source, **kw):
            return [_ResultWithBoxes()]

    # Inject into cache so constructor uses our fake
    YOLODetector._model_cache["yolov8n"] = _YOLOWithBoxes("yolov8n")
    det = YOLODetector(model_name="yolov8n", device="cpu")

    img = np.zeros((480, 640, 3), dtype="uint8")
    result = det.detect_image(img, annotate=False)

    assert len(result.detections) == 2
    assert result.detections[0].class_name == "person"
    assert result.detections[1].class_name == "car"


def test_detect_image_inference_ms_positive():
    det = YOLODetector(model_name="yolov8n", device="cpu")
    img = np.zeros((480, 640, 3), dtype="uint8")
    result = det.detect_image(img, annotate=False)
    assert result.inference_ms >= 0.0
