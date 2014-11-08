#!/usr/bin/env python

import cv2
import numpy as np
from shoe_monster.msg import Target


class Shoe(object):
    #States for selecting the region of interest
    SELECTING_ROI_PT_1 = 0
    SELECTING_ROI_PT_2 = 1
    ROI_SELECTED = 2

    def __init__(self, img):
        self.query_img = img
        self.state = Shoe.SELECTING_ROI_PT_1

        cv2.namedWindow("OBJSEL")
        cv2.setMouseCallback("OBJSEL",self.mouse_event)
        cv2.imshow("OBJSEL",self.query_img)

        #wait to select the ROI
        while True:
            if self.state != Shoe.ROI_SELECTED:
                cv2.waitKey(100)
            else:
                break
        cv2.destroyWindow("OBJSEL")

        self.get_query_histogram()

    def scale_img(self, test_img, shoe_img):
        '''Scale shoe_img to match test_img, in case those are different
        (doesn't matter right now)
        '''
        scale,dim = max([(float(test_img.shape[i])/shoe_img.shape[i],i) for i in [0,1]])
        shoe_img = np.array(cv2.resize(shoe_img,(int(shoe_img.shape[1]*scale),int(shoe_img.shape[0]*scale))))
        crop_dim = 1-dim
        crop_min = (shoe_img.shape[crop_dim]-test_img.shape[crop_dim])/2
        crop_max = crop_min+test_img.shape[crop_dim]
        if crop_dim == 1:
            shoe_img = shoe_img[:,crop_min:crop_max]
        else:
            shoe_img = shoe_img[crop_min:crop_max,:]
        return shoe_img

    def mouse_event(self,event,x,y,flag,im):
        '''code for selecting the region of interest
        runs on mouse click
        '''
        if event == cv2.EVENT_FLAG_LBUTTON:
            if self.state == Shoe.SELECTING_ROI_PT_1:
                self.query_roi = [x,y,-1,-1]
                # cv2.circle(img.query_img_visualize,(x,y),5,(255,0,0),5)
                self.state = Shoe.SELECTING_ROI_PT_2
            elif self.state == Shoe.SELECTING_ROI_PT_2:
                self.query_roi[2:] = [x,y]
                self.last_detection = self.query_roi
                # cv2.circle(img.query_img_visualize,(x,y),5,(255,0,0),5)
                self.state = Shoe.ROI_SELECTED
                self.get_query_histogram()

    def get_query_histogram(self):
        '''calculate the histogram to try to match
        '''
        # set up the ROI for tracking
        roi = self.query_img[self.query_roi[1]:self.query_roi[3],self.query_roi[0]:self.query_roi[2],:]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # play with the number of histogram bins by changing histSize
        self.query_hist = cv2.calcHist([hsv_roi],[0],mask=None,histSize=[256],ranges=[0,255])
        cv2.normalize(self.query_hist,self.query_hist,0,255,cv2.NORM_MINMAX)

    def find_center(self,im):
        '''actually do the tracking!'''
        im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        track_im = cv2.calcBackProject([im_hsv],[0],self.query_hist,[0,255],1)

        track_im_visualize = track_im.copy()
        # convert to (x,y,w,h)
        track_roi = (self.last_detection[0],self.last_detection[1],self.last_detection[2]-self.last_detection[0],self.last_detection[3]-self.last_detection[1])
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        # this is done to plot intermediate results of mean shift
        for max_iter in range(1,10):
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1 )
            (ret, intermediate_roi) = cv2.meanShift(track_im,track_roi,term_crit)
            cv2.rectangle(track_im_visualize,(intermediate_roi[0],intermediate_roi[1]),(intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]),max_iter/10.0,2)

        # find average of secion of track_im 2,3 is width and height for prob stuff
        
        self.last_detection = [intermediate_roi[0],intermediate_roi[1],intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]]
        #get the center of the box
        posX = (self.last_detection[0]+self.last_detection[2])/2
        posY = (self.last_detection[1]+self.last_detection[3])/2

        cv2.circle(im,(posX,posY),2,(255,0,0),10)
        cv2.imshow('image',im)
        cv2.waitKey(20)

        return Target(x = posX, y = posY, x_img_size = self.query_img.shape[0],y_img_size = self.query_img.shape[1])
