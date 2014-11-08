#!/usr/bin/env python

import numpy as np
import cv2
import track_red as track
import rospy

from track_shoe import Shoe


from sensor_msgs.msg import Image
from std_msgs.msg import String
from shoe_monster.msg import Target
from cv_bridge import CvBridge, CvBridgeError

class ObjectTracking:
    def __init__(self):
        self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.track_object)
        self.target_pub = rospy.Publisher("target", Target)
        self.bridge = CvBridge()
        self.identifier = None
        self.i = 0
        print "initiated"

    def track_object(self,msg):

        # Bridge for color image. This page was very useful for deteermaning image types: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        print "hello"
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = np.asanyarray(cv_image)
        if not self.identifier:
            self.i += 1
            if self.i>50:
                #shoe_img = cv2.imread('/home/rboy/catkin_ws/src/shoe_monster/images/shoe2_right.jpg')
                self.identifier = Shoe(image)

        if self.identifier:
            prob, obj_center = self.identifier.find_center(image)
            if prob > .8:
                self.target_pub.publish(obj_center)
            else:
                self.target_pub.publish(Target(x = -1, y = -1, x_img_size = -1,y_img_size = -1))

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



