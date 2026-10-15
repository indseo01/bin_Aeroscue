import cv2
import time
import os
import numpy as np
from ultralytics import YOLO


# ============================================================
#                    AEROSCUE CONFIGURATION
# ============================================================

HUMAN_MODEL_PATH = "yolov8_human.engine"
FLOOD_MODEL_PATH = "yolov8_flood.engine"

# Your phone IP camera
IP_CAMERA_URL = "http://192.168.1.134:8080/video"

CONF_THRESHOLD = 0.50

WINDOW_NAME = "AEROSCUE - Human + Flood Detection"

# Number of consecutive failed reads before giving up
MAX_FRAME_FAILURES = 10

# Small delay between failed read attempts
FRAME_RETRY_DELAY = 0.10


# ============================================================
#                    LOAD MODELS
# ============================================================

print("\nLoading models...")

try:
    human_model = YOLO(HUMAN_MODEL_PATH, task="detect")
    flood_model = YOLO(FLOOD_MODEL_PATH, task="obb")

    print("Models loaded successfully.")

except Exception as e:
    print("\nERROR: Could not load YOLO models.")
    print("Reason:", e)
    raise


# ============================================================
#                    OPEN IP CAMERA
# ============================================================

source = IP_CAMERA_URL

print("\nOpening IP camera:")
print(source)

cap = cv2.VideoCapture(source)

# Try to minimize buffering
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

if not cap.isOpened():
    print("\nERROR: Could not open IP camera.")
    print("Check:")
    print("1. Phone and Jetson are on the same Wi-Fi.")
    print("2. IP address is correct.")
    print("3. IP Camera app is running.")
    print("4. Port 8080 is accessible.")
    raise SystemExit(1)


# ============================================================
#                 READ FIRST FRAME
# ============================================================

print("Waiting for first frame...")

first_frame = None

for attempt in range(MAX_FRAME_FAILURES):

    ret, frame = cap.read()

    if ret and frame is not None:
        first_frame = frame
        break

    print(
        f"Waiting for camera frame "
        f"({attempt + 1}/{MAX_FRAME_FAILURES})..."
    )

    time.sleep(FRAME_RETRY_DELAY)


if first_frame is None:
    print("\nERROR: Could not receive a frame from IP camera.")

    cap.release()

    raise SystemExit(1)


# ============================================================
#                  CAMERA INFORMATION
# ============================================================

height, width = first_frame.shape[:2]

source_fps = cap.get(cv2.CAP_PROP_FPS)

if source_fps <= 0 or np.isnan(source_fps):
    source_fps = 0.0


# ============================================================
#                  CREATE WINDOW
# ============================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Resize displayed window without changing inference resolution
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


# ============================================================
#                     WARMUP
# ============================================================

print("\nWarming up YOLO models...")

try:

    # Human warmup
    human_model.predict(
        first_frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    # Flood warmup
    flood_model.predict(
        first_frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    print("Warmup completed.")

except Exception as e:

    print("\nWARNING: Warmup error:")
    print(e)

    print("Continuing anyway...")


# ============================================================
#                     BENCHMARK DATA
# ============================================================

total_frames = 0

total_human_detections = 0
total_flood_detections = 0

human_times = []
flood_times = []
combined_ai_times = []
frame_times = []
fps_history = []

benchmark_start = time.perf_counter()

last_frame_time = benchmark_start

consecutive_failures = 0

exit_reason = "Unknown"


# ============================================================
#                 DETECTION LOOP
# ============================================================

print("\n============================================================")
print("              AEROSCUE LIVE DETECTION")
print("============================================================")
print("Input                 : IP Camera")
print("Resolution            :", width, "x", height)
print("Source FPS            :", f"{source_fps:.2f}")
print()
print("Press Q to quit.")
print("Press Ctrl+C to quit safely.")
print("============================================================\n")


try:

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        frame_start = time.perf_counter()

        # ====================================================
        #                  READ FRAME
        # ====================================================

        ret, frame = cap.read()

        if not ret or frame is None:

            consecutive_failures += 1

            print(
                f"\nWARNING: Could not read frame "
                f"({consecutive_failures}/{MAX_FRAME_FAILURES})"
            )

            if consecutive_failures >= MAX_FRAME_FAILURES:

                print("\nERROR: IP camera stream lost.")

                exit_reason = "IP camera stream lost"

                break

            time.sleep(FRAME_RETRY_DELAY)

            continue

        # Successful frame
        consecutive_failures = 0

        # ====================================================
        #               HUMAN DETECTION
        # ====================================================

        human_start = time.perf_counter()

        try:

            human_results = human_model.predict(
                frame,
                conf=CONF_THRESHOLD,
                verbose=False
            )

        except Exception as e:

            print("\nERROR during human detection:")
            print(e)

            exit_reason = "Human inference error"

            break

        human_end = time.perf_counter()

        human_latency = (
            human_end - human_start
        ) * 1000.0

        human_times.append(human_latency)


        # ====================================================
        #               FLOOD DETECTION
        # ====================================================

        flood_start = time.perf_counter()

        try:

            flood_results = flood_model.predict(
                frame,
                conf=CONF_THRESHOLD,
                verbose=False
            )

        except Exception as e:

            print("\nERROR during flood detection:")
            print(e)

            exit_reason = "Flood inference error"

            break

        flood_end = time.perf_counter()

        flood_latency = (
            flood_end - flood_start
        ) * 1000.0

        flood_times.append(flood_latency)


        # ====================================================
        #                 AI LATENCY
        # ====================================================

        combined_ai_latency = (
            human_latency +
            flood_latency
        )

        combined_ai_times.append(combined_ai_latency)


        # ====================================================
        #                 DRAW HUMAN BOXES
        # ====================================================

        human_count = 0

        if human_results:

            result = human_results[0]

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(len(boxes)):

                    confidence = float(
                        boxes.conf[i].item()
                    )

                    if confidence < CONF_THRESHOLD:
                        continue

                    human_count += 1

                    xyxy = boxes.xyxy[i].cpu().numpy()

                    x1, y1, x2, y2 = map(
                        int,
                        xyxy
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"Person "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )


        # ====================================================
        #                 DRAW FLOOD OBB
        # ====================================================

        flood_count = 0

        if flood_results:

            result = flood_results[0]

            # OBB results
            if result.obb is not None:

                obb = result.obb

                for i in range(len(obb)):

                    confidence = float(
                        obb.conf[i].item()
                    )

                    if confidence < CONF_THRESHOLD:
                        continue

                    flood_count += 1

                    # xyxyxyxy gives 4 corner points
                    points = (
                        obb.xyxyxyxy[i]
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )

                    points = points.reshape(
                        (-1, 1, 2)
                    )

                    cv2.polylines(
                        frame,
                        [points],
                        True,
                        (0, 0, 255),
                        3
                    )

                    # Center for label
                    center_x = int(
                        np.mean(points[:, 0, 0])
                    )

                    center_y = int(
                        np.mean(points[:, 0, 1])
                    )

                    label = (
                        f"Flood "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )


        # ====================================================
        #                  UPDATE COUNTS
        # ====================================================

        total_human_detections += human_count
        total_flood_detections += flood_count

        total_frames += 1


        # ====================================================
        #                FRAME LATENCY
        # ====================================================

        frame_end = time.perf_counter()

        frame_latency = (
            frame_end - frame_start
        ) * 1000.0

        frame_times.append(frame_latency)


        # ====================================================
        #                     FPS
        # ====================================================

        current_fps = (
            1000.0 / frame_latency
            if frame_latency > 0
            else 0.0
        )

        fps_history.append(current_fps)


        # ====================================================
        #                LIVE OVERLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (500, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Human: {human_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Flood: {flood_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {current_fps:.2f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"AI: {combined_ai_latency:.1f} ms",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        #                    DISPLAY
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        # ====================================================
        #                    KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):

            print("\nQ pressed. Stopping benchmark...")

            exit_reason = "User pressed Q"

            break


# ============================================================
#                 CTRL+C HANDLING
# ============================================================

except KeyboardInterrupt:

    print("\n\nCtrl+C detected.")

    print("Stopping safely...")

    exit_reason = "User pressed Ctrl+C"


# ============================================================
#                 UNEXPECTED ERROR
# ============================================================

except Exception as e:

    print("\n\n============================================================")
    print("UNEXPECTED ERROR")
    print("============================================================")

    print(type(e).__name__)
    print(str(e))

    exit_reason = (
        "Unexpected Python exception"
    )


# ============================================================
#                       CLEANUP
# ============================================================

finally:

    print("\nCleaning up...")

    try:
        cap.release()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# ============================================================
#                  FINAL BENCHMARK
# ============================================================

benchmark_end = time.perf_counter()

total_runtime = (
    benchmark_end -
    benchmark_start
)


# ------------------------------------------------------------
# Avoid division by zero
# ------------------------------------------------------------

if total_frames > 0 and total_runtime > 0:

    actual_fps = (
        total_frames /
        total_runtime
    )

else:

    actual_fps = 0.0


# ------------------------------------------------------------
# Average latency
# ------------------------------------------------------------

if human_times:
    avg_human = np.mean(human_times)
else:
    avg_human = 0.0


if flood_times:
    avg_flood = np.mean(flood_times)
else:
    avg_flood = 0.0


if combined_ai_times:
    avg_ai = np.mean(combined_ai_times)
else:
    avg_ai = 0.0


if frame_times:
    avg_frame = np.mean(frame_times)
else:
    avg_frame = 0.0


# ------------------------------------------------------------
# Min / Max frame latency
# ------------------------------------------------------------

if frame_times:

    min_frame = np.min(frame_times)
    max_frame = np.max(frame_times)

else:

    min_frame = 0.0
    max_frame = 0.0


# ------------------------------------------------------------
# FPS statistics
# ------------------------------------------------------------

if fps_history:

    min_fps = np.min(fps_history)
    max_fps = np.max(fps_history)

    # 1% low
    sorted_fps = np.sort(
        np.array(fps_history)
    )

    one_percent_count = max(
        1,
        int(len(sorted_fps) * 0.01)
    )

    one_percent_low = np.mean(
        sorted_fps[:one_percent_count]
    )

else:

    min_fps = 0.0
    max_fps = 0.0
    one_percent_low = 0.0


# ------------------------------------------------------------
# Theoretical AI FPS
# ------------------------------------------------------------

if avg_ai > 0:

    theoretical_ai_fps = (
        1000.0 / avg_ai
    )

else:

    theoretical_ai_fps = 0.0


# ------------------------------------------------------------
# Theoretical complete frame FPS
# ------------------------------------------------------------

if avg_frame > 0:

    theoretical_frame_fps = (
        1000.0 / avg_frame
    )

else:

    theoretical_frame_fps = 0.0


# ------------------------------------------------------------
# Pipeline overhead
# ------------------------------------------------------------

pipeline_overhead = (
    avg_frame -
    avg_ai
)

if pipeline_overhead < 0:
    pipeline_overhead = 0.0


# ============================================================
#                    PRINT BENCHMARK
# ============================================================

print("\n")
print("=" * 60)
print("                 AEROSCUE BENCHMARK")
print("=" * 60)


print("\n---------------- SYSTEM ----------------")

print(
    f"Input                 : IP Camera"
)

print(
    f"Stream URL            : {IP_CAMERA_URL}"
)

print(
    f"Resolution            : "
    f"{width} x {height}"
)

print(
    f"Source FPS            : "
    f"{source_fps:.2f}"
)


print("\n---------------- EXECUTION -------------")

print(
    f"Frames processed       : "
    f"{total_frames}"
)

print(
    f"Runtime                : "
    f"{total_runtime:.2f} sec"
)

print(
    f"Exit reason            : "
    f"{exit_reason}"
)


print("\n---------------- LATENCY ---------------")

print(
    f"Human inference       : "
    f"{avg_human:.2f} ms"
)

print(
    f"Flood inference        : "
    f"{avg_flood:.2f} ms"
)

print(
    f"Combined AI            : "
    f"{avg_ai:.2f} ms"
)

print(
    f"Frame processing       : "
    f"{avg_frame:.2f} ms"
)

print(
    f"Pipeline overhead      : "
    f"{pipeline_overhead:.2f} ms"
)


print("\n---------------- PERFORMANCE ------------")

print(
    f"Actual average FPS     : "
    f"{actual_fps:.2f}"
)

print(
    f"Minimum FPS            : "
    f"{min_fps:.2f}"
)

print(
    f"Maximum FPS            : "
    f"{max_fps:.2f}"
)

print(
    f"1% low FPS             : "
    f"{one_percent_low:.2f}"
)

print(
    f"Theoretical AI FPS     : "
    f"{theoretical_ai_fps:.2f}"
)

print(
    f"Theoretical frame FPS  : "
    f"{theoretical_frame_fps:.2f}"
)


print("\n---------------- FRAME LATENCY ---------")

print(
    f"Minimum frame time     : "
    f"{min_frame:.2f} ms"
)

print(
    f"Maximum frame time     : "
    f"{max_frame:.2f} ms"
)


print("\n---------------- DETECTIONS ------------")

print(
    f"Total human detections : "
    f"{total_human_detections}"
)

print(
    f"Total flood detections : "
    f"{total_flood_detections}"
)


print("\n============================================================")
print("                 BENCHMARK COMPLETE")
print("============================================================")
