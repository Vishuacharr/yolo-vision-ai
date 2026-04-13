"""
CLI entrypoint for YOLO Vision AI.

Examples:
    python src/detect.py --source 0 --model yolov8n
    python src/detect.py --source video.mp4 --model yolov8m --save
    python src/detect.py --source image.jpg --model yolov8s --show
"""

import argparse
import sys
from pathlib import Path

import cv2

from detector import YOLODetector


def parse_args():
    p = argparse.ArgumentParser(description="YOLO Vision AI — Object Detection CLI")
    p.add_argument("--source", required=True,
                   help="Input source: 0 (webcam), path to image/video, or URL")
    p.add_argument("--model", default="yolov8n",
                   choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--show", action="store_true", help="Display live output")
    p.add_argument("--save", action="store_true", help="Save annotated output")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    detector = YOLODetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Webcam
    if args.source.isdigit():
        detector.stream_webcam(source=int(args.source), show=True)
        return

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"[ERROR] Source not found: {args.source}")
        sys.exit(1)

    suffix = src_path.suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if is_video:
        cap = cv2.VideoCapture(str(src_path))
        writer = None
        if args.save:
            out_path = src_path.with_stem(src_path.stem + "_detected")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.detect_image(frame, annotate=True)
            frame_count += 1
            print(f"\rFrame {frame_count}: {len(result.detections)} detections | "
                  f"{result.inference_ms:.1f} ms", end="")

            if args.show and result.annotated_frame is not None:
                cv2.imshow("YOLO Vision AI", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer and result.annotated_frame is not None:
                writer.write(result.annotated_frame)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n[✓] Processed {frame_count} frames")

    else:
        # Image
        result = detector.detect_from_file(src_path, annotate=True)
        print(f"[✓] {len(result.detections)} objects detected in {result.inference_ms:.1f} ms")
        for det in result.detections:
            print(f"  • {det.class_name}: {det.confidence:.2%}  bbox={det.bbox}")

        if args.show and result.annotated_frame is not None:
            cv2.imshow("YOLO Vision AI", result.annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if args.save and result.annotated_frame is not None:
            out_path = src_path.with_stem(src_path.stem + "_detected")
            cv2.imwrite(str(out_path), result.annotated_frame)
            print(f"[✓] Saved to {out_path}")


if __name__ == "__main__":
    main()
