from ultralytics import YOLO
import time

IMAGE_PATH = "test1.jpg"
CONF = 0.5
ITERATIONS = 10

print("Loading PyTorch model...")
pt_model = YOLO("yolov8_human.pt", task="detect")

print("Loading TensorRT model...")
trt_model = YOLO("yolov8_human.engine", task="detect")

# --------------------------------------------------
# PyTorch benchmark
# --------------------------------------------------
print("\nBenchmarking PyTorch...")

# Warm-up
for _ in range(3):
    pt_model(IMAGE_PATH, conf=CONF, verbose=False)

start = time.time()

for _ in range(ITERATIONS):
    pt_model(IMAGE_PATH, conf=CONF, verbose=False)

pt_time = (time.time() - start) / ITERATIONS



# --------------------------------------------------
# TensorRT benchmark
# --------------------------------------------------
print("\nBenchmarking TensorRT...")

# Warm-up
for _ in range(3):
    trt_model(IMAGE_PATH, conf=CONF, verbose=False)

start = time.time()

for _ in range(ITERATIONS):
    trt_model(IMAGE_PATH, conf=CONF, verbose=False)

trt_time = (time.time() - start) / ITERATIONS



# --------------------------------------------------
# Comparison
# --------------------------------------------------
speedup = pt_time / trt_time

print("\n==============================")
print("       BENCHMARK RESULT")
print("==============================")
print(f"PyTorch :  {pt_time * 1000:.1f} ms")
print(f"TensorRT:  {trt_time * 1000:.1f} ms")
print(f"PyTorch FPS:     {1 / pt_time:.2f}")
print(f"TensorRT FPS:     {1 / trt_time:.2f}")
print(f"TensorRT average: {trt_time * 1000:.1f} ms")
print(f"PyTorch average: {pt_time * 1000:.1f} ms")
print(f"Speedup  :  {speedup:.2f}x")
print("==============================")
