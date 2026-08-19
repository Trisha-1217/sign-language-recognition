import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Model path
MODEL_PATH = "models/hand_landmarker.task"


# Create Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)

print("Starting webcam...")
print("Press Q to quit.")


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Flip frame for natural webcam view
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = detector.detect(mp_image)

    # Print landmark coordinates
    if result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):

            print(f"\nHand {hand_index + 1}")

            for landmark_index, landmark in enumerate(hand_landmarks):

                print(
                    f"Landmark {landmark_index}: "
                    f"x={landmark.x:.4f}, "
                    f"y={landmark.y:.4f}, "
                    f"z={landmark.z:.4f}"
                )

    # Display webcam
    cv2.imshow("Hand Landmark Detection", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()