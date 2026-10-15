"""
Aeroscue - Combined Human + Flood Detection

Runs:
1. YOLO human detection model
2. YOLO flood OBB (Oriented Bounding Box) model

Both models run on the same image and the results
are displayed together.
"""

from ultralytics import YOLO
import cv2
import os


# ============================================================
# CONFIGURATION
# ============================================================

CONF_THRESHOLD = 0.3

HUMAN_MODEL_PATH = "yolov8_human.pt"
FLOOD_MODEL_PATH = "yolov8_flood.pt"


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

human_model = YOLO(HUMAN_MODEL_PATH)
flood_model = YOLO(FLOOD_MODEL_PATH)

print("Models loaded.\n")


# ============================================================ 	
# RUN DETECTION
# ============================================================

def run_detection(image_path):

    results = {
        "humans": [],
        "flood": []
    }

    # --------------------------------------------------------
    # HUMAN DETECTION
    # --------------------------------------------------------

    human_results = human_model(
        image_path,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    human_result = human_results[0]

    if human_result.boxes is not None:

        for box in human_result.boxes:

            conf = float(box.conf[0])

            xyxy = box.xyxy[0].tolist()

            results["humans"].append({
                "confidence": conf,
                "box": xyxy
            })


    # --------------------------------------------------------
    # FLOOD DETECTION
    # OBB MODEL
    # --------------------------------------------------------

    flood_results = flood_model(
        image_path,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    flood_result = flood_results[0]

    # OBB detections
    if flood_result.obb is not None:

        for i in range(len(flood_result.obb)):

            conf = float(
                flood_result.obb.conf[i]
            )

            # Four corner points
            corners = (
                flood_result
                .obb
                .xyxyxyxy[i]
                .cpu()
                .numpy()
                .tolist()
            )

            results["flood"].append({
                "confidence": conf,
                "corners": corners
            })


    return results, human_results, flood_results


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results, image_path):

    print(
        f"--- Detection Summary for {image_path} ---"
    )

    # --------------------------------------------------------
    # HUMANS
    # --------------------------------------------------------

    print(
        f"People detected: {len(results['humans'])}"
    )

    for i, human in enumerate(
        results["humans"],
        1
    ):

        print(
            f"  Person {i}: "
            f"confidence "
            f"{human['confidence']:.2f}"
        )


    # --------------------------------------------------------
    # FLOOD
    # --------------------------------------------------------

    print(
        f"Flood zones detected: "
        f"{len(results['flood'])}"
    )

    for i, flood in enumerate(
        results["flood"],
        1
    ):

        print(
            f"  Flood {i}: "
            f"confidence "
            f"{flood['confidence']:.2f}"
        )


    print("-------------------------------------------")
    print()


# ============================================================
# DISPLAY COMBINED IMAGE
# ============================================================

def show_combined_image(
    human_results,
    flood_results
):

    # --------------------------------------------------------
    # START WITH HUMAN DETECTIONS
    # --------------------------------------------------------

    annotated = human_results[0].plot()


    # --------------------------------------------------------
    # DRAW FLOOD OBB DETECTIONS
    # --------------------------------------------------------

    flood_result = flood_results[0]

    if flood_result.obb is not None:

        for i in range(
            len(flood_result.obb)
        ):

            # Get four corner points
            corners = (
                flood_result
                .obb
                .xyxyxyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )

            # Confidence
            conf = float(
                flood_result
                .obb
                .conf[i]
            )


            # ------------------------------------------------
            # DRAW FLOOD POLYGON
            # ------------------------------------------------

            cv2.polylines(
                annotated,
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
                annotated,
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


    else:

        print(
            "No flood zones detected "
            "above the confidence threshold."
        )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "Aeroscue - Combined Detection",
        annotated
    )

    print(
        "Press any key on the image window "
        "to close it."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # ASK FOR IMAGE
    # --------------------------------------------------------

    print("Aeroscue Detection System")
    print("==========================")
    print()

    IMAGE_PATH = input(
        "Enter the image path: "
    ).strip().strip('"')


    # --------------------------------------------------------
    # CHECK IMAGE EXISTS
    # --------------------------------------------------------

    if not os.path.isfile(IMAGE_PATH):

        print()
        print("ERROR: Image file not found.")
        print(
            f"Path entered: {IMAGE_PATH}"
        )
        print()

        exit()


    # --------------------------------------------------------
    # CHECK IMAGE CAN BE READ
    # --------------------------------------------------------

    test_image = cv2.imread(IMAGE_PATH)

    if test_image is None:

        print()
        print(
            "ERROR: OpenCV could not read "
            "the image."
        )
        print()

        exit()


    # --------------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------------

    print()
    print(
        f"Processing image: {IMAGE_PATH}"
    )
    print()


    results, human_results, flood_results = (
        run_detection(IMAGE_PATH)
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print_summary(
        results,
        IMAGE_PATH
    )


    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    show_combined_image(
        human_results,
        flood_results
    )
