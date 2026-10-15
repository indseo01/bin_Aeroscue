"""
============================================================
AEROSCUE - HUMAN + FLOOD LIVE DETECTION + BENCHMARK
============================================================

Models:
1. YOLO Human Detection - TensorRT
2. YOLO Flood OBB Detection - TensorRT

Input:
- USB camera
- Video file

Display:
- Human bounding boxes
- Flood OBB polygons
- Human count
- Flood count
- Current FPS
- Average FPS
- Human inference latency
- Flood inference latency
- Combined AI latency
- End-to-end frame latency

Controls:
- Q       -> Quit
- Ctrl+C  -> Quit safely and print benchmark

Final benchmark:
- Total frames
- Runtime
- Actual average FPS
- Minimum FPS
- Maximum FPS
- 1% low FPS
- Human latency
- Flood latency
- Combined AI latency
- End-to-end latency
- Pipeline overhead
- Theoretical AI FPS
- Total detections

============================================================
"""

from ultralytics import YOLO
import cv2
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

CONF_THRESHOLD = 0.3

HUMAN_MODEL_PATH = "yolov8_human.engine"
FLOOD_MODEL_PATH = "yolov8_flood.engine"

IMAGE_SIZE = 640

WINDOW_NAME = "AeroScue - Human + Flood"


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("============================================================")
print("       AEROSCUE - HUMAN + FLOOD LIVE DETECTION")
print("============================================================")
print()

print("Loading Human TensorRT model...")

human_model = YOLO(
    HUMAN_MODEL_PATH,
    task="detect"
)

print("Human model loaded.")
print()

print("Loading Flood TensorRT OBB model...")

flood_model = YOLO(
    FLOOD_MODEL_PATH,
    task="obb"
)

print("Flood model loaded.")
print()

print("Both TensorRT models loaded successfully.")
print()


# ============================================================
# SELECT INPUT
# ============================================================

source = input(
    "Enter camera index (0) or video path: "
).strip().strip('"')


# ============================================================
# DETERMINE CAMERA OR VIDEO
# ============================================================

if source.isdigit():

    source = int(source)

    print()
    print(f"Opening camera index: {source}")

else:

    if not os.path.isfile(source):

        print()
        print("ERROR: Video file not found.")
        print(f"Path: {source}")
        print()

        raise SystemExit


    print()
    print(f"Opening video: {source}")


# ============================================================
# OPEN CAMERA / VIDEO
# ============================================================

cap = cv2.VideoCapture(source)

if not cap.isOpened():

    print()
    print("ERROR: Could not open camera/video.")
    print()

    raise SystemExit


# ============================================================
# CAMERA SETTINGS
# ============================================================

if isinstance(source, int):

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )


# ============================================================
# SOURCE INFORMATION
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

source_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print()
print("Source information")
print("------------------")
print(f"Resolution : {width} x {height}")
print(f"Source FPS : {source_fps:.2f}")
print()


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, frame = cap.read()

if not ret:

    print("ERROR: Could not read first frame.")

    cap.release()

    raise SystemExit


# ============================================================
# CREATE RESIZABLE WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)


# ============================================================
# WARM-UP
# ============================================================

print("Warming up TensorRT engines...")

for _ in range(5):

    human_model(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    flood_model(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

print("Warm-up complete.")
print()


# ============================================================
# STATISTICS
# ============================================================

total_frames = 0

total_human_time = 0.0
total_flood_time = 0.0
total_ai_time = 0.0
total_frame_time = 0.0

total_human_detections = 0
total_flood_detections = 0

# FPS history
fps_history = []

# Frame latency history
frame_latency_history = []

# Rolling FPS
fps_frame_count = 0

fps_timer = time.perf_counter()

display_fps = 0.0

# Benchmark timer
benchmark_start = time.perf_counter()

# Used to distinguish normal exit from Ctrl+C
stopped_by_keyboard = False


# ============================================================
# MAIN DETECTION LOOP
# ============================================================

print("============================================================")
print("             LIVE DETECTION STARTED")
print("============================================================")
print()
print("Press Q to quit.")
print("Press Ctrl+C to quit safely.")
print()


try:

    while True:

        # ====================================================
        # START FRAME TIMER
        # ====================================================

        frame_start = time.perf_counter()


        # ====================================================
        # READ FRAME
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            if isinstance(source, str):

                print()
                print("Video finished.")

            else:

                print()
                print("Camera frame could not be read.")

            break


        # ====================================================
        # HUMAN DETECTION
        # ====================================================

        human_start = time.perf_counter()

        human_results = human_model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        human_time = (
            time.perf_counter()
            - human_start
        ) * 1000


        # ====================================================
        # FLOOD DETECTION
        # ====================================================

        flood_start = time.perf_counter()

        flood_results = flood_model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        flood_time = (
            time.perf_counter()
            - flood_start
        ) * 1000


        # ====================================================
        # COMBINED AI LATENCY
        # ====================================================

        ai_time = (
            human_time
            + flood_time
        )


        # ====================================================
        # HUMAN RESULT
        # ====================================================

        human_result = human_results[0]

        if human_result.boxes is not None:

            human_count = len(
                human_result.boxes
            )

        else:

            human_count = 0


        # ====================================================
        # FLOOD RESULT
        # ====================================================

        flood_result = flood_results[0]

        if flood_result.obb is not None:

            flood_count = len(
                flood_result.obb
            )

        else:

            flood_count = 0


        # ====================================================
        # CREATE DISPLAY FRAME
        # ====================================================

        display_frame = frame.copy()


        # ====================================================
        # DRAW HUMAN DETECTIONS
        # ====================================================

        if human_result.boxes is not None:

            for box in human_result.boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                conf = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = xyxy

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"HUMAN {conf:.2f}",
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # DRAW FLOOD OBB DETECTIONS
        # ====================================================

        if flood_result.obb is not None:

            for i in range(
                len(flood_result.obb)
            ):

                corners = (
                    flood_result
                    .obb
                    .xyxyxyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                conf = float(
                    flood_result
                    .obb
                    .conf[i]
                )

                cv2.polylines(
                    display_frame,
                    [corners],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=3
                )

                x, y = corners[0]

                cv2.putText(
                    display_frame,
                    f"FLOOD {conf:.2f}",
                    (
                        int(x),
                        max(int(y) - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )


        # ====================================================
        # END-TO-END FRAME LATENCY
        # ====================================================

        frame_time = (
            time.perf_counter()
            - frame_start
        ) * 1000


        # ====================================================
        # UPDATE STATISTICS
        # ====================================================

        total_frames += 1

        total_human_time += human_time

        total_flood_time += flood_time

        total_ai_time += ai_time

        total_frame_time += frame_time

        total_human_detections += human_count

        total_flood_detections += flood_count

        frame_latency_history.append(
            frame_time
        )


        # ====================================================
        # FPS
        # ====================================================

        fps_frame_count += 1

        current_time = time.perf_counter()

        elapsed = (
            current_time
            - fps_timer
        )


        if elapsed >= 1.0:

            display_fps = (
                fps_frame_count
                / elapsed
            )

            fps_history.append(
                display_fps
            )

            fps_frame_count = 0

            fps_timer = current_time


        # ====================================================
        # RUNNING AVERAGES
        # ====================================================

        avg_frame_live = (
            total_frame_time
            / total_frames
        )

        running_fps = (
            1000.0
            / avg_frame_live
        )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            display_frame,
            (10, 10),
            (440, 280),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # TITLE
        # ====================================================

        cv2.putText(
            display_frame,
            "AEROSCUE",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )


        # ====================================================
        # CURRENT FPS
        # ====================================================

        cv2.putText(
            display_frame,
            f"FPS: {display_fps:.2f}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )


        # ====================================================
        # RUNNING FPS
        # ====================================================

        cv2.putText(
            display_frame,
            f"Avg FPS: {running_fps:.2f}",
            (180, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )


        # ====================================================
        # HUMAN COUNT
        # ====================================================

        cv2.putText(
            display_frame,
            f"Humans: {human_count}",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


        # ====================================================
        # FLOOD COUNT
        # ====================================================

        cv2.putText(
            display_frame,
            f"Flood zones: {flood_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )


        # ====================================================
        # HUMAN LATENCY
        # ====================================================

        cv2.putText(
            display_frame,
            f"Human AI: {human_time:.1f} ms",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )


        # ====================================================
        # FLOOD LATENCY
        # ====================================================

        cv2.putText(
            display_frame,
            f"Flood AI: {flood_time:.1f} ms",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )


        # ====================================================
        # AI TOTAL
        # ====================================================

        cv2.putText(
            display_frame,
            f"AI Total: {ai_time:.1f} ms",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )


        # ====================================================
        # FRAME LATENCY
        # ====================================================

        cv2.putText(
            display_frame,
            f"Frame: {frame_time:.1f} ms",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )


        # ====================================================
        # FRAME COUNT
        # ====================================================

        cv2.putText(
            display_frame,
            f"Frames: {total_frames}",
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )


        # ====================================================
        # KEYBOARD CONTROL
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print()
            print("Q pressed. Stopping benchmark...")

            break


except KeyboardInterrupt:

    stopped_by_keyboard = True

    print()
    print()
    print("Ctrl+C detected.")
    print("Stopping safely and generating benchmark...")


finally:

    # ========================================================
    # STOP TIMER
    # ========================================================

    benchmark_end = time.perf_counter()

    total_runtime = (
        benchmark_end
        - benchmark_start
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# FINAL BENCHMARK
# ============================================================

print()
print()
print("============================================================")
print("                 AEROSCUE BENCHMARK")
print("============================================================")


if total_frames > 0:

    # ========================================================
    # AVERAGE LATENCIES
    # ========================================================

    avg_human = (
        total_human_time
        / total_frames
    )

    avg_flood = (
        total_flood_time
        / total_frames
    )

    avg_ai = (
        total_ai_time
        / total_frames
    )

    avg_frame = (
        total_frame_time
        / total_frames
    )


    # ========================================================
    # ACTUAL FPS
    # ========================================================

    actual_fps = (
        total_frames
        / total_runtime
    )


    # ========================================================
    # THEORETICAL FPS
    # ========================================================

    theoretical_ai_fps = (
        1000.0
        / avg_ai
    )

    theoretical_frame_fps = (
        1000.0
        / avg_frame
    )


    # ========================================================
    # MIN / MAX FPS
    # ========================================================

    if len(fps_history) > 0:

        min_fps = min(
            fps_history
        )

        max_fps = max(
            fps_history
        )

    else:

        min_fps = 0.0
        max_fps = 0.0


    # ========================================================
    # 1% LOW FPS
    # ========================================================

    if len(fps_history) >= 5:

        sorted_fps = sorted(
            fps_history
        )

        one_percent_count = max(
            1,
            int(
                len(sorted_fps)
                * 0.01
            )
        )

        one_percent_low = (
            sum(
                sorted_fps[
                    :one_percent_count
                ]
            )
            / one_percent_count
        )

    else:

        one_percent_low = min_fps


    # ========================================================
    # PIPELINE OVERHEAD
    # ========================================================

    pipeline_overhead = (
        avg_frame
        - avg_ai
    )


    # ========================================================
    # PRINT SYSTEM INFO
    # ========================================================

    print()
    print("---------------- SYSTEM ----------------")

    if isinstance(source, int):

        print(
            f"Input                 : "
            f"Camera {source}"
        )

    else:

        print(
            f"Input                 : Video"
        )

    print(
        f"Resolution            : "
        f"{width} x {height}"
    )

    print(
        f"Source FPS            : "
        f"{source_fps:.2f}"
    )


    # ========================================================
    # EXECUTION
    # ========================================================

    print()
    print("---------------- EXECUTION -------------")

    print(
        f"Frames processed       : "
        f"{total_frames}"
    )

    print(
        f"Runtime                : "
        f"{total_runtime:.2f} sec"
    )


    # ========================================================
    # LATENCY
    # ========================================================

    print()
    print("---------------- LATENCY ---------------")

    print(
        f"Human inference       : "
        f"{avg_human:.2f} ms"
    )

    print(
        f"Flood inference       : "
        f"{avg_flood:.2f} ms"
    )

    print(
        f"Combined AI           : "
        f"{avg_ai:.2f} ms"
    )

    print(
        f"End-to-end frame      : "
        f"{avg_frame:.2f} ms"
    )

    print(
        f"Pipeline overhead     : "
        f"{pipeline_overhead:.2f} ms"
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    print()
    print("---------------- PERFORMANCE -----------")

    print(
        f"Actual average FPS    : "
        f"{actual_fps:.2f}"
    )

    print(
        f"Minimum FPS           : "
        f"{min_fps:.2f}"
    )

    print(
        f"Maximum FPS           : "
        f"{max_fps:.2f}"
    )

    print(
        f"1% low FPS            : "
        f"{one_percent_low:.2f}"
    )

    print(
        f"Theoretical AI FPS    : "
        f"{theoretical_ai_fps:.2f}"
    )

    print(
        f"Theoretical frame FPS : "
        f"{theoretical_frame_fps:.2f}"
    )


    # ========================================================
    # DETECTIONS
    # ========================================================

    print()
    print("---------------- DETECTIONS ------------")

    print(
        f"Total human detections: "
        f"{total_human_detections}"
    )

    print(
        f"Total flood detections: "
        f"{total_flood_detections}"
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("============================================================")
    print("                 BENCHMARK COMPLETE")
    print("============================================================")

    if stopped_by_keyboard:

        print(
            "Stopped using Ctrl+C."
        )

    else:

        print(
            "Stopped normally."
        )


else:

    print()
    print("No frames were processed.")


print()
print("AeroScue test finished.")
