import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in grayscale
img = cv2.imread('redball.jpg',1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

bottom_red = np.array([0,50,50])
lower_red = np.array([2,255,255])
upper_red = np.array([160,50,50])
top_red = np.array([179,255,255])

mask1 = cv2.inRange(hsv, bottom_red, lower_red)
mask2 = cv2.inRange(hsv, upper_red, top_red)
mask = cv2.bitwise_or(mask1, mask2)

M = cv2.moments(mask)

posX = int(M['m10']/M['m00'])
posY = int(M['m01']/M['m00'])

cv2.circle(img,(posX,posY),2,(255,0,0))

res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow('mask',mask)
cv2.imshow('image',img)
cv2.imshow('res',res)
k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()


