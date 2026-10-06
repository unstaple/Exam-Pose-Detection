from pathlib import Path
import sys
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

VIDEO = ROOT / "Videos" / "7092083-hd_1920_1080_30fps.mp4"
MODEL = ROOT / "Models" / "yolov8n-face.pt"

print("=" * 60, flush=True)
print("SMOKE TEST", flush=True)
print("=" * 60, flush=True)

print(f"Python: {sys.version}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(
        f"CUDA device: {torch.cuda.get_device_name(0)}",
        flush=True,
    )

print(f"Video: {VIDEO}", flush=True)
print(f"Video exists: {VIDEO.exists()}", flush=True)

print(f"YOLO model: {MODEL}", flush=True)
print(f"Model exists: {MODEL.exists()}", flush=True)

if not VIDEO.exists():
    raise FileNotFoundError(VIDEO)

if not MODEL.exists():
    raise FileNotFoundError(MODEL)

print("\nOpening video...", flush=True)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("OpenCV could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolution: {width} x {height}", flush=True)
print(f"FPS: {fps}", flush=True)
print(f"Frame count: {frames}", flush=True)

print("\nLoading YOLO...", flush=True)

model = YOLO(str(MODEL))

print("YOLO loaded.", flush=True)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame.")

print("First frame read.", flush=True)

print("Running YOLO detection...", flush=True)

results = model.predict(
    source=frame,
    conf=0.35,
    imgsz=1280,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    verbose=False,
)

print(
    f"Detection complete: {len(results[0].boxes)} boxes",
    flush=True,
)

annotated = results[0].plot()

cv2.imshow(
    "Smoke Test",
    annotated,
)

print(
    "\nPress any key in the OpenCV window to exit.",
    flush=True,
)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print("Smoke test passed.", flush=True)