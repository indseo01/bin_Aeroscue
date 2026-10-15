"""
============================================================
AEROSCUE - COMBINED HUMAN + FLOOD LIVE DETECTION
============================================================

Runs:
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
- Human inference latency
- Flood inference latency
- Combined AI latency
- End-to-end frame latency
- FPS

Controls:
- Q -> Quit

The OpenCV display window is RESIZABLE.
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
print("AEROSCUE - HUMAN + FLOOD LIVE DETECTION")
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

        exit()

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

    exit()


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

    print(
        "ERROR: Could not read first frame."
    )

    cap.release()

    exit()


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

fps_frame_count = 0

fps_timer = time.perf_counter()

display_fps = 0.0


# ============================================================
# START LIVE DETECTION
# ============================================================

print("============================================================")
print("LIVE DETECTION STARTED")
print("============================================================")
print()
print("Press Q to quit.")
print()


while True:

    # ========================================================
    # START FRAME TIMER
    # ========================================================

    frame_start = time.perf_counter()


    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        # ----------------------------------------------------
        # VIDEO FILE ENDED
        # ----------------------------------------------------

        if isinstance(source, str):

            print()
            print("Video finished.")

            break

        # ----------------------------------------------------
        # CAMERA FAILURE
        # ----------------------------------------------------

        else:

            print()
            print("Camera frame could not be read.")

            break


    # ========================================================
    # HUMAN DETECTION
    # ========================================================

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


    # ========================================================
    # FLOOD DETECTION
    # ========================================================

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


    # ========================================================
    # COMBINED AI LATENCY
    # ========================================================

    ai_time = (
        human_time
        + flood_time
    )


    # ========================================================
    # HUMAN RESULT
    # ========================================================

    human_result = human_results[0]


    if human_result.boxes is not None:

        human_count = len(
            human_result.boxes
        )

    else:

        human_count = 0


    # ========================================================
    # CREATE DISPLAY FRAME
    # ========================================================

    display_frame = frame.copy()


    # ========================================================
    # DRAW HUMAN DETECTIONS
    # ========================================================

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


            # ------------------------------------------------
            # HUMAN BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # HUMAN LABEL
            # ------------------------------------------------

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


    # ========================================================
    # FLOOD RESULT
    # ========================================================

    flood_result = flood_results[0]


    if flood_result.obb is not None:

        flood_count = len(
            flood_result.obb
        )

    else:

        flood_count = 0


    # ========================================================
    # DRAW FLOOD OBB DETECTIONS
    # ========================================================

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


            # ------------------------------------------------
            # FLOOD POLYGON
            # ------------------------------------------------

            cv2.polylines(
                display_frame,
                [corners],
                isClosed=True,
                color=(255, 0, 0),
                thickness=3
            )


            # ------------------------------------------------
            # FLOOD LABEL
            # ------------------------------------------------

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


    # ========================================================
    # END-TO-END FRAME LATENCY
    # ========================================================

    frame_time = (
        time.perf_counter()
        - frame_start
    ) * 1000


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    total_frames += 1

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

        fps_frame_count = 0

        fps_timer = current_time


    # ========================================================
    # ACCUMULATE STATISTICS
    # ========================================================

    total_human_time += human_time

    total_flood_time += flood_time

    total_ai_time += ai_time

    total_frame_time += frame_time


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        display_frame,
        (10, 10),
        (390, 205),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"FPS: {display_fps:.2f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # HUMAN COUNT
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"Humans: {human_count}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # FLOOD COUNT
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"Flood zones: {flood_count}",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 0, 0),
        2
    )


    # --------------------------------------------------------
    # HUMAN LATENCY
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"Human: {human_time:.1f} ms",
        (20, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # FLOOD LATENCY
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"Flood: {flood_time:.1f} ms",
        (20, 153),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # COMBINED AI LATENCY
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"AI total: {ai_time:.1f} ms",
        (20, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # END-TO-END LATENCY
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"Frame: {frame_time:.1f} ms",
        (200, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD CONTROL
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("============================================================")
print("AEROSCUE LIVE TEST COMPLETE")
print("============================================================")


if total_frames > 0:

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


    print()

    print(
        f"Frames processed       : "
        f"{total_frames}"
    )

    print(
        f"Average Human latency  : "
        f"{avg_human:.2f} ms"
    )

    print(
        f"Average Flood latency  : "
        f"{avg_flood:.2f} ms"
    )

    print(
        f"Average AI latency     : "
        f"{avg_ai:.2f} ms"
    )

    print(
        f"Average Frame latency  : "
        f"{avg_frame:.2f} ms"
    )

    print(
        f"Theoretical AI FPS     : "
        f"{1000 / avg_ai:.2f}"
    )

    print(
        f"Theoretical Frame FPS  : "
        f"{1000 / avg_frame:.2f}"
    )


print()
print("AeroScue test finished.")
