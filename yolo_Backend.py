import cv2
from ultralytics import YOLO
import time
import os
import csv

# Load model
model = YOLO("YOLO-Weights/bestest.pt")

# Allowed classes from your model
ALLOWED = {
    2: "Hardhat",
    11: "Safety Vest",
    5: "NO-Hardhat",
    7: "NO-Safety Vest"
}

# For saving violation screenshots
last_saved = {}  # key = track_id, value = timestamp

# Make sure violations folder exists
os.makedirs("violations", exist_ok=True)

# CSV log file for violations
LOG_FILE = "violations_log.csv"

# Ensure CSV has header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "class_name", "track_id", "image_path"])


def log_violation(class_name, track_id, image_path):
    """Append a violation entry to the CSV log."""
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), class_name, track_id, image_path])


def detect_frame(frame, conf_thres):
    results = model.track(frame, persist=True, conf=conf_thres)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in ALLOWED:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            class_name = ALLOWED[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Colors
            if cls == 2:        # Hardhat
                color = (0, 255, 255)
            elif cls == 11:     # Safety Vest
                color = (0, 255, 0)
            elif cls == 5:      # NO-Hardhat
                color = (0, 0, 255)
            elif cls == 7:      # NO-Safety Vest
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)

            # Draw bounding box
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


            # SAVE VIOLATION (ONLY ONCE)
      
            if "NO-" in class_name and track_id is not None:
                now = time.time()

                # First time for this track_id
                if track_id not in last_saved:
                    filename = f"violations/{class_name}_{track_id}.jpg"
                    cv2.imwrite(filename, frame)
                    last_saved[track_id] = now
                    log_violation(class_name, track_id, filename)

                # Optional: resave only after 8 seconds
                elif now - last_saved[track_id] > 8:
                    filename = f"violations/{class_name}_{track_id}_{int(now)}.jpg"
                    cv2.imwrite(filename, frame)
                    last_saved[track_id] = now
                    log_violation(class_name, track_id, filename)

    return frame
