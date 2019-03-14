import numpy as np
import cv2

cap = cv2.VideoCapture(0)

beginCapture = False
i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if beginCapture:
        print('Start capturing')
        if (i % 2 == 0):
            cv2.imwrite(str(int(i/2)) + '.png', frame)
        i = i + 1
    if beginCapture and i >= 100:
        print('Stop caturing')
        beginCapture = False

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        beginCapture = True
        print('Start capturing')
        i = 0

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()