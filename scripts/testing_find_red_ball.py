import numpy as np
import cv2
import track_red as track


def find_red_center(img, convert_hsv=True):
    if convert_hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bottom_red = np.array([0,50,50])
    lower_red = np.array([2,255,255])
    upper_red = np.array([160,50,50])
    top_red = np.array([179,255,255])

    mask1 = cv2.inRange(img, bottom_red, lower_red)
    mask2 = cv2.inRange(img, upper_red, top_red)
    mask = cv2.bitwise_or(mask1, mask2)

    #consider doing erode/dilate cycle to clean up object

    M = cv2.moments(mask)

    posX = int(M['m10']/M['m00'])
    posY = int(M['m01']/M['m00'])

    return posX, posY


if __name__ == "__main__":

    cap = cv2.VideoCapture(-1)
    tracker = track.RedTracker()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        posX, posY = tracker.find_red_center(frame)

        # Our operations on the frame come here
        cv2.circle(frame,(posX,posY),2,(255,0,0),5)

        cv2.imshow('image',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



