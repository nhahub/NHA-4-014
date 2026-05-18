import argparse
import json
import os
import struct
import sys
import time
import traceback

FRAME_STREAM = sys.stdout.buffer
sys.stdout = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detector frame streamer for the WPF app.")
    parser.add_argument("--model", help="Path to a .pt model. Omit to stream frames without detection.")
    parser.add_argument("--list-classes", action="store_true", help="Print model class names as JSON and exit.")
    parser.add_argument("--source-mode", choices=("camera", "video"))
    parser.add_argument("--source", help="Camera index or video path.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def open_capture(cv2, source_mode: str, source: str):
    status(f"Opening {source_mode} source...")
    if source_mode == "camera":
        capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def write_frame(frame_bytes: bytes) -> None:
    FRAME_STREAM.write(struct.pack("<I", len(frame_bytes)))
    FRAME_STREAM.write(frame_bytes)
    FRAME_STREAM.flush()


def write_json(value) -> None:
    FRAME_STREAM.write(json.dumps(value).encode("utf-8"))
    FRAME_STREAM.flush()


def model_classes(model) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]

    return [str(name) for name in names]


def main() -> int:
    args = parse_args()

    try:
        import cv2
    except ModuleNotFoundError as exc:
        status(f"Missing Python package: {exc.name}. Run: python -m pip install -r requirements.txt")
        return 3

    if args.model and not os.path.exists(args.model):
        status(f"Model not found: {args.model}")
        return 4

    if args.list_classes:
        if not args.model:
            write_json([])
            return 0

        from ultralytics import YOLO
        model = YOLO(args.model)
        write_json(model_classes(model))
        return 0

    if not args.source_mode or args.source is None:
        status("source-mode and source are required for detection.")
        return 2

    cap = open_capture(cv2, args.source_mode, args.source)
    if not cap.isOpened():
        status(f"Could not open {args.source_mode} source: {args.source}")
        return 5

    model = None
    if args.model:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            status(f"Missing Python package: {exc.name}. Run: python -m pip install -r requirements.txt")
            return 3

        status("Loading YOLO model...")
        model = YOLO(args.model)
    else:
        status("No model selected. Streaming frames without detection.")

    target_interval = 0.0 if args.fps <= 0 else 1.0 / args.fps
    next_frame_time = 0.0
    frame_count = 0
    last_status_time = time.perf_counter()

    fps_label = "maximum available FPS" if args.fps <= 0 else f"{args.fps:g} FPS"
    status(f"Stream is running at {fps_label}.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.source_mode == "video":
                    status("Video finished.")
                    return 0

                time.sleep(0.05)
                continue

            if target_interval > 0:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)

                next_frame_time = time.perf_counter() + target_interval

            if model is not None:
                results = model.predict(
                    frame,
                    imgsz=args.image_size,
                    conf=args.confidence,
                    verbose=False,
                )
                frame = results[0].plot()

            success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                status("Could not encode detector frame.")
                continue

            write_frame(encoded.tobytes())
            frame_count += 1

            if time.perf_counter() - last_status_time >= 5.0:
                status(f"Running. Processed {frame_count} frames.")
                last_status_time = time.perf_counter()
    finally:
        cap.release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception:
        status(traceback.format_exc())
        raise SystemExit(10)
