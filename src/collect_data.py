import cv2
import csv
import os
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/hand_landmarker.task"

DATASET_DIR = "dataset/landmarks"
CSV_PATH = os.path.join(DATASET_DIR, "landmarks.csv")


# ============================================================
# CREATE DATASET FOLDER
# ============================================================

os.makedirs(DATASET_DIR, exist_ok=True)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# CSV SETUP
# ============================================================

header = ["label"]

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


file_exists = os.path.exists(CSV_PATH)

csv_file = open(
    CSV_PATH,
    "a",
    newline=""
)

writer = csv.writer(csv_file)

if not file_exists or os.path.getsize(CSV_PATH) == 0:
    writer.writerow(header)
    csv_file.flush()


# ============================================================
# ASK FOR SIGN LABEL
# ============================================================

label = input(
    "Enter sign label (example: A): "
).strip().upper()

if not label:
    print("No label entered. Exiting.")
    csv_file.close()
    detector.close()
    exit()


print()
print("=" * 50)
print(f"Collecting data for sign: {label}")
print("=" * 50)
print("SPACE = Save sample")
print("Q     = Quit")
print("ESC   = Quit")
print()
print("IMPORTANT: Click the webcam window before pressing keys.")
print()


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    csv_file.close()
    detector.close()
    exit()


sample_count = 0
timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp
    timestamp_ms += 33

    # Run MediaPipe Hand Landmarker
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Sign: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Save | Q/ESC = Quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DRAW HAND LANDMARKS
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Draw landmarks
        for landmark in hand:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        for start, end in connections:

            x1 = int(
                hand[start].x * frame.shape[1]
            )

            y1 = int(
                hand[start].y * frame.shape[0]
            )

            x2 = int(
                hand[end].x * frame.shape[1]
            )

            y2 = int(
                hand[end].y * frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # ========================================================
    # SHOW WEBCAM
    # ========================================================

    cv2.imshow(
        "Sign Language Data Collection",
        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(10) & 0xFF


    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    if key == 32:  # SPACE

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            row = [label]

            # 21 landmarks × 3 coordinates = 63 values
            for landmark in hand:

                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            writer.writerow(row)
            csv_file.flush()

            sample_count += 1

            print(
                f"Saved sample {sample_count} "
                f"for sign {label}"
            )

        else:

            print(
                "No hand detected. "
                "Put your hand inside the camera."
            )


    # ========================================================
    # QUIT
    # ========================================================

    if key == ord("q") or key == ord("Q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

csv_file.close()

detector.close()

cv2.destroyAllWindows()


print()
print("=" * 50)
print("DATA COLLECTION FINISHED")
print("=" * 50)
print(f"Sign: {label}")
print(f"Samples collected: {sample_count}")
print(f"Dataset saved to: {CSV_PATH}")
print()