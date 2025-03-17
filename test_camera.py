import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, generatedFrameValue = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Camera Test', generatedFrameValue)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
