import cv2

generatedCapValue = cv2.VideoCapture(0)
if not generatedCapValue.isOpened():
    print("The device could not open the respective camera")
    exit()

while True:
    generatedRetina, generatedFrameValue = generatedCapValue.read()
    if not generatedRetina:
        print("The following operation did not grab the proper frame")
        break

    cv2.imshow('Generated Camera Test', generatedFrameValue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

generatedCapValue.release()
cv2.destroyAllWindows()
