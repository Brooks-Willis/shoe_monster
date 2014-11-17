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

        # List of identifiers to consider when tracking the object
        # identifiers should implement AbstractTracker from track_shoe
        self.identifiers = [HistTracker(),KeypointTracker()]

        self.i = 0
        self.state = self.SELECTING_QUERY_IMG
        self.query_roi=None
        self.query_img = None
        self.current_frame = None
        self.current_center = None

        cv2.namedWindow("Chase the Shoe")
        cv2.setMouseCallback("Chase the Shoe",self.mouse_event)


    def show_img(self):
        '''vizualise the image with the tracking data'''
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
                # draw where we think the center is
                cv2.circle(combined_img,self.current_center,2,(255,0,0),10)

                cv2.imshow("Chase the Shoe",combined_img)
            else:
                cv2.imshow("Chase the Shoe",self.current_frame)
        else:
            cv2.imshow("Chase the Shoe",self.query_img_visualize)
        cv2.waitKey(20)

    def weighted_average(self, probs, vals):
        tot_p = sum(probs)
        tot_v = sum([vals[i]*probs[i] for i in range(len(vals))])
        return float(tot_v)/tot_p

    def track_object(self,msg):
        '''do the object tracking!'''

        # Bridge for color image. This page was very useful for deteermaning image types: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = np.asanyarray(cv_image)
        self.current_frame = image

        objs = []
        probs = []
        for idr in self.identifiers:
            # get the hypothesized center from every tracker
            prob, obj_center = idr.track(image.copy(),viz=True)
            if prob > .7:  #only consider probs over 70%
                probs.append(prob)
                objs.append(obj_center)
        if len(objs)>0:
            # take the weighted average center
            # alternative: take the most probable center
            # (that's probably better, honestly. Why didn't we do that?)
            xs,ys = zip(*objs)
            out = Target(x = int(self.weighted_average(probs,xs)),
                         y = int(self.weighted_average(probs,ys)),
                         x_img_size=self.query_img.shape[0],
                         y_img_size=self.query_img.shape[1])
        else:
            out = Target(x=-1,y=-1,x_img_size=-1,y_img_size=-1)

        self.current_center = out.x,out.y
        self.target_pub.publish(out)
        self.show_img()
        

    def mouse_event(self,event,x,y,flag,im):
        '''select the region of interest! Yeah, we took this pretty much directly
        from your stuff, Paul'''
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