"""
Aeroscue - Combined Human + Flood Detection
Runs both YOLOv8 models on the same image and shows combined results.
"""

from ultralytics import YOLO
import cv2

# --- Config ---
IMAGE_PATH = "test1.jpg"      # change this to whatever image you're testing
CONF_THRESHOLD = 0.5         # minimum confidence to count as a real detection
HUMAN_MODEL_PATH = "yolov8_human.pt"
FLOOD_MODEL_PATH = "yolov8_flood.pt"

# --- Load both models once ---
print("Loading models...")
human_model = YOLO(HUMAN_MODEL_PATH)
flood_model = YOLO(FLOOD_MODEL_PATH)
print("Models loaded.\n")


def run_detection(image_path):
    """Run both models on one image and return combined detection info."""
    results = {"humans": [], "flood": []}

    # --- Human detection ---
    human_results = human_model(image_path, conf=CONF_THRESHOLD, verbose=False)
    for box in human_results[0].boxes:
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        results["humans"].append({"confidence": conf, "box": xyxy})

    # --- Flood detection (OBB model: oriented bounding boxes, not regular boxes) ---
    flood_results = flood_model(image_path, conf=CONF_THRESHOLD, verbose=False)
    if flood_results[0].obb is not None:
        for i in range(len(flood_results[0].obb)):
            conf = float(flood_results[0].obb.conf[i])
            corners = flood_results[0].obb.xyxyxyxy[i].tolist()  # 4 corner points
            results["flood"].append({"confidence": conf, "corners": corners})

    return results, human_results, flood_results


def print_summary(results):
    print(f"--- Detection Summary for {IMAGE_PATH} ---")
    print(f"People detected:  {len(results['humans'])}")
    for i, h in enumerate(results["humans"], 1):
        print(f"  Person {i}: confidence {h['confidence']:.2f}")

    print(f"Flood zones detected: {len(results['flood'])}")
    for i, f in enumerate(results["flood"], 1):
        print(f"  Flood {i}: confidence {f['confidence']:.2f}")
    print("-------------------------------------------\n")


def show_combined_image(human_results, flood_results):
    """Overlay both models' boxes on one image and display it."""
    # Start from the human model's annotated frame
    annotated = human_results[0].plot()

    # Draw flood boxes on top in a different color (blue) so they're distinguishable
    for box in flood_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, f"flood {conf:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Aeroscue - Combined Detection", annotated)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    results, human_results, flood_results = run_detection(IMAGE_PATH)
    print_summary(results)
    show_combined_image(human_results, flood_results)
