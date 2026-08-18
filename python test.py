import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imshow("Camera test",frame)
    if cv2.waitkey(1) == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()






