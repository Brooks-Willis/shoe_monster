import numpy as np
import cv2

class RedTracker(object):

    def __init__(self, default_conversion=cv2.COLOR_BGR2HSV):
        self.conversion= default_conversion

        bottom_red = np.array([0,50,50])
        lower_red = np.array([2,255,255])
        upper_red = np.array([160,50,50])
        top_red = np.array([179,255,255])

    def mask(self, img):
        img = cv2.cvtColor(img, self.conversion)

        bottom_red = np.array([0,50,50])
        lower_red = np.array([2,255,255])
        upper_red = np.array([160,50,50])
        top_red = np.array([179,255,255])

        mask1 = cv2.inRange(img, bottom_red, lower_red)
        mask2 = cv2.inRange(img, upper_red, top_red)
        mask = cv2.bitwise_or(mask1, mask2)

        #consider doing erode/dilate cycle to clean up object

        return mask

    def find_center(self,contour):

        M = cv2.moments(contour)

        posX = int(M['m10']/M['m00'])
        posY = int(M['m01']/M['m00'])

        return posX, posY

    def find_red_center(self,img):

        mask = self.mask(img)
        return self.find_center(mask)

