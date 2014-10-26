#!/usr/bin/env python

import numpy as np
import cv2
import track_red as track
import rospy


from sensor_msgs.msg import Image
from std_msgs.msg import String
from shoe_monster.msg import target
from cv_bridge import CvBridge, CvBridgeError

class ObjectTracking:
    def __init__(self):
        self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.track_object)
        self.target_pub = rospy.Publisher("target", target)
        self.bridge = CvBridge()
        print "initiated"

    def track_object(self,msg):

        cv_image = self.bridge.imgmsg_to_cv(msg, "16UC1")
        #cv_image = self.bridge.imgmsg_to_cv(msg, "rbg8")
        image = np.asanyarray(cv_image)

        identifier = RedBall()

        print identifier.find_center(image)

class RedBall:
    def __init__(self):
        pass

    def find_center(img, convert_hsv=True):
        print "finding center"
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
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        cv2.imshow('post-processed',dilation)

        M = cv2.moments(dilation)

        posX = int(M['m10']/M['m00'])
        posY = int(M['m01']/M['m00'])

        self.target_pub()
        return posX, posY


if __name__ == "__main__":

    rospy.init_node('track_object', anonymous=True)
    object_tracker = ObjectTracking()
    rospy.spin()
    # cap = cv2.VideoCapture(-1)
    # tracker = track.RedTracker()

    # while(True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()

    #     posX, posY = tracker.find_red_center(frame)

    #     # Our operations on the frame come here
    #     cv2.circle(frame,(posX,posY),2,(255,0,0),5)

    #     cv2.imshow('image',frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()



