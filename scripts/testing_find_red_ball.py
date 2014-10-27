#!/usr/bin/env python

import numpy as np
import cv2
import track_red as track
import rospy


from sensor_msgs.msg import Image
from std_msgs.msg import String
from shoe_monster.msg import Target
from cv_bridge import CvBridge, CvBridgeError

class ObjectTracking:
    def __init__(self):
        self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.track_object)
        self.target_pub = rospy.Publisher("target", Target)
        self.bridge = CvBridge()
        print "initiated"

    def track_object(self,msg):

        # Bridge for color image. This page was very useful for deteermaning image types: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = np.asanyarray(cv_image)
        identifier = RedBall(image)

        obj_center = identifier.find_center(image)

        self.target_pub.publish(obj_center)

class RedBall:
    def __init__(self,image):
        # Determine size of image
        self.image_size = image.shape

    def find_center(self, image, convert_hsv=True):
        if convert_hsv:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        bottom_red = np.array([0,50,50])
        lower_red = np.array([10,255,255])
        upper_red = np.array([160,50,50])
        top_red = np.array([179,255,255])

        mask1 = cv2.inRange(img, bottom_red, lower_red)
        mask2 = cv2.inRange(img, upper_red, top_red)
        mask = cv2.bitwise_or(mask1, mask2)

        #consider doing erode/dilate cycle to clean up object
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)

        M = cv2.moments(dilation)

        if M['m00'] >= 10: # Only determine average moment when it detects noticible red object
            posX = int(M['m10']/M['m00'])
            posY = int(M['m01']/M['m00'])

        else: # No object found
            posX = -1
            posY = -1

        cv2.circle(image,(posX,posY),2,(255,0,0),10)
        cv2.imshow('post-processed',dilation)
        cv2.imshow('image',image)
        cv2.waitKey(20)

        return Target(x = posX, y = posY, x_img_size = self.image_size[0],y_img_size = self.image_size[1])

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



