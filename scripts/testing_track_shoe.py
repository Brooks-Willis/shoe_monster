#!/usr/bin/env python

import numpy as np
import cv2
import track_red as track
import rospy

from track_shoe import HistTracker, KeypointTracker


from sensor_msgs.msg import Image
from std_msgs.msg import String
from shoe_monster.msg import Target
from cv_bridge import CvBridge, CvBridgeError

class ObjectTracking:
    SELECTING_QUERY_IMG = 0
    SELECTING_ROI_PT_1 = 1
    SELECTING_ROI_PT_2 = 2

    def __init__(self):
        self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.track_object)
        self.target_pub = rospy.Publisher("target", Target)
        self.bridge = CvBridge()
        self.identifiers = [HistTracker(), KeypointTracker()]
        self.i = 0
        self.state = self.SELECTING_QUERY_IMG
        self.query_roi=None
        self.query_img = None
        self.current_frame = None
        self.current_center = None

        cv2.namedWindow("Chase the Shoe")
        cv2.setMouseCallback("Chase the Shoe",self.mouse_event)

        print "initiated"

    def show_img(self):
        if self.state == self.SELECTING_QUERY_IMG:
            if self.query_roi != None: #Runs after selecting inital roi
                # add the query image to the side
                combined_img = np.zeros((self.current_frame.shape[0],
                                         self.current_frame.shape[1]+(self.query_roi[2]-self.query_roi[0]),
                                         self.current_frame.shape[2]),
                                        dtype=self.current_frame.dtype)
                combined_img[:,0:self.current_frame.shape[1],:] = self.current_frame
                combined_img[0:(self.query_roi[3]-self.query_roi[1]),self.current_frame.shape[1]:,:] = (
                        self.query_img[self.query_roi[1]:self.query_roi[3],
                                          self.query_roi[0]:self.query_roi[2],:])
                print self.current_center
                cv2.circle(combined_img,self.current_center,2,(255,0,0),10)

                cv2.imshow("Chase the Shoe",combined_img)
            else:
                cv2.imshow("Chase the Shoe",self.current_frame)
        else:
            cv2.imshow("Chase the Shoe",self.query_img_visualize)
        cv2.waitKey(20)


    def track_object(self,msg):

        # Bridge for color image. This page was very useful for deteermaning image types: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        print "hello"
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = np.asanyarray(cv_image)
        self.current_frame = image

        objs = []
        for idr in self.identifiers:
            prob, obj_center = idr.track(image.copy())
            if prob > .7:
                objs.append(obj_center)
        if len(objs)>0:
            xs,ys = zip(*objs)
            out = Target(x=int(np.mean(xs)),
                         y = int(np.mean(ys)),
                         x_img_size=self.query_img.shape[0],
                         y_img_size=self.query_img.shape[1])
        else:
            out = Target(x=-1,y=-1,x_img_size=-1,y_img_size=-1)
        print prob
        self.current_center = out.x,out.y
        print self.current_center
        print self.current_frame.shape
        self.target_pub.publish(out)
        self.show_img()
        

    def mouse_event(self,event,x,y,flag,im):
        if event == cv2.EVENT_FLAG_LBUTTON:
            if self.state == self.SELECTING_QUERY_IMG:
                self.query_img_visualize = self.current_frame.copy()
                self.query_img = self.current_frame
                self.query_roi = None
                self.state = self.SELECTING_ROI_PT_1
            elif self.state == self.SELECTING_ROI_PT_1:
                self.query_roi = [x,y,-1,-1]
                cv2.circle(self.query_img_visualize,(x,y),5,(255,0,0),5)
                self.state = self.SELECTING_ROI_PT_2
            else:
                self.query_roi[2:] = [x,y]
                cv2.circle(self.query_img_visualize,(x,y),5,(255,0,0),5)
                self.state = self.SELECTING_QUERY_IMG
                for idr in self.identifiers:
                    idr.set_query(self.query_img,self.query_roi)

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