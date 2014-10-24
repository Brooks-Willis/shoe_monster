import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

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

	cv2.imshow('mask',mask)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 2)
	dilation = cv2.dilate(erosion,kernel,iterations = 1)
	cv2.imshow('post-processed',dilation)

	M = cv2.moments(dilation)

	posX = int(M['m10']/M['m00'])
	posY = int(M['m01']/M['m00'])

	return posX, posY

if __name__ == "__main__":
	for filename in os.listdir("RedBalls"):
		img = cv2.imread(os.getcwd()+"/RedBalls/"+filename)
		posX, posY = find_red_center(img)

		cv2.circle(img,(posX,posY),2,(255,0,0),5)

		cv2.imshow('image',img)

		k = cv2.waitKey(0) & 0xFF
		if k == 27:         # wait for ESC key to exit
			 cv2.destroyAllWindows()
		elif k == ord('s'): # wait for 's' key to save and exit
			cv2.imwrite('messigray.png',img)
			cv2.destroyAllWindows()