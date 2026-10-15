from ultralytics import YOLO
import time

IMAGE_PATH = "test1.jpg"
ITERATIONS = 50
WARMUP = 10

pt_model = YOLO("yolov8_flood.pt", task="obb")
trt_model = YOLO("yolov8_flood.engine", task="obb")

# -------------------------------
# PyTorch warm-up
# -------------------------------
for _ in range(WARMUP):
    pt_model(IMAGE_PATH, conf=0.5, verbose=False)

# -------------------------------
# PyTorch benchmark
# -------------------------------
start = time.perf_counter()

for _ in range(ITERATIONS):
    pt_model(IMAGE_PATH, conf=0.5, verbose=False)

pt_time = (time.perf_counter() - start) / ITERATIONS


# -------------------------------
# TensorRT warm-up
# -------------------------------
for _ in range(WARMUP):
    trt_model(IMAGE_PATH, conf=0.5, verbose=False)

# -------------------------------
# TensorRT benchmark
# -------------------------------
start = time.perf_counter()

for _ in range(ITERATIONS):
    trt_model(IMAGE_PATH, conf=0.5, verbose=False)

trt_time = (time.perf_counter() - start) / ITERATIONS


# -------------------------------
# Results
# -------------------------------
print("\n========== flood MODEL ==========")
print(f"PyTorch :  {pt_time * 1000:.2f} ms")
print(f"TensorRT:  {trt_time * 1000:.2f} ms")
print(f"PyTorch FPS :  {1 / pt_time:.2f}")
print(f"TensorRT FPS:  {1 / trt_time:.2f}")
print(f"TensorRT average: {trt_time * 1000:.1f} ms")
print(f"PyTorch average: {pt_time * 1000:.1f} ms")
print(f"Speedup: {pt_time / trt_time:.2f}x")
print("=================================")
