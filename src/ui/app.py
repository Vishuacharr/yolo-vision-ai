"""
Streamlit UI for YOLO Vision AI.
Supports: image upload, webcam stream, video file, URL.
"""

import base64
import io
import time

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="YOLO Vision AI",
    page_icon="👁️",
    layout="wide",
)

API_URL = "http://localhost:8000"

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.image("https://ultralytics.com/images/logo.svg", width=150)
st.sidebar.title("⚙️ Settings")

model = st.sidebar.selectbox("Model", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                             help="Larger = more accurate, slower")
conf = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.05)
source = st.sidebar.radio("Input source", ["📷 Upload Image", "🌐 Image URL", "📹 Webcam"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with YOLOv8 + FastAPI**")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
st.title("👁️ YOLO Vision AI")
st.markdown("Real-time object detection powered by **YOLOv8** and **Ultralytics**")

col_input, col_output = st.columns(2)


def call_api(image_bytes: bytes) -> dict:
    img_b64 = base64.b64encode(image_bytes).decode()
    resp = requests.post(f"{API_URL}/detect", json={
        "image_b64": img_b64,
        "model": model,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "return_annotated": True,
    })
    resp.raise_for_status()
    return resp.json()


def display_result(result: dict):
    detections = result.get("detections", [])
    st.success(f"✅ Found **{len(detections)}** objects in **{result['inference_ms']:.1f} ms**")

    if result.get("annotated_image_b64"):
        img_bytes = base64.b64decode(result["annotated_image_b64"])
        st.image(img_bytes, caption="Annotated output", use_column_width=True)

    if detections:
        st.subheader("📊 Detections")
        st.dataframe(
            [{"Class": d["class_name"], "Confidence": f"{d['confidence']:.2%}",
              "BBox": str(d["bbox"])} for d in detections],
            use_container_width=True,
        )


if source == "📷 Upload Image":
    with col_input:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            st.image(uploaded, caption="Input image", use_column_width=True)

    if uploaded and st.button("🚀 Detect", type="primary"):
        with st.spinner("Running inference..."):
            try:
                result = call_api(uploaded.read())
                with col_output:
                    display_result(result)
            except Exception as e:
                st.error(f"API error: {e}")

elif source == "🌐 Image URL":
    with col_input:
        url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        if url:
            st.image(url, caption="Input image", use_column_width=True)

    if url and st.button("🚀 Detect", type="primary"):
        with st.spinner("Downloading and running inference..."):
            try:
                img_data = requests.get(url, timeout=10).content
                result = call_api(img_data)
                with col_output:
                    display_result(result)
            except Exception as e:
                st.error(f"Error: {e}")

elif source == "📹 Webcam":
    st.info("📹 Webcam mode runs locally. Make sure the API server is running.")
    if st.button("▶️ Start Webcam Detection"):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        stop_btn = st.button("⏹️ Stop")
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode(".jpg", frame)
            try:
                result = call_api(buf.tobytes())
                if result.get("annotated_image_b64"):
                    img = base64.b64decode(result["annotated_image_b64"])
                    stframe.image(img, channels="BGR", use_column_width=True)
            except Exception:
                stframe.image(frame, channels="BGR", use_column_width=True)
            time.sleep(0.03)
        cap.release()
