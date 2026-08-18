import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"


# Create MediaPipe Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        print("\n--- Hand Landmarks ---")

        for i, landmark in enumerate(hand):

            print(
                f"Landmark {i}: "
                f"x={landmark.x:.4f}, "
                f"y={landmark.y:.4f}, "
                f"z={landmark.z:.4f}"
            )

    cv2.imshow(
        "Landmark Data",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()