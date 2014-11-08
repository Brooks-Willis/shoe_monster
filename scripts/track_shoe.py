#!/usr/bin/env python

import cv2
import numpy as np
from shoe_monster.msg import Target


class Shoe(object):
    #States for selecting the region of interest
    SELECTING_ROI = 0
    ROI_SELECTED = 1

    def __init__(self):
        self.query_img = None
        self.query_roi = None
        self.state = self.SELECTING_ROI

        self.confidence_count = 0

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

    def set_query(self,img,roi):
        '''ROI should be [x_topleft,y_topleft,x_bottomright,y_bottomright]'''
        self.query_img = img
        self.query_roi = roi
        self.get_query_histogram()
        self.last_detection = roi
        self.state = self.ROI_SELECTED

    def get_query_histogram(self):
        '''calculate the histogram to try to match
        '''
        # set up the ROI for tracking
        roi = self.query_img[self.query_roi[1]:self.query_roi[3],self.query_roi[0]:self.query_roi[2],:]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # play with the number of histogram bins by changing histSize
        self.query_hist = cv2.calcHist([hsv_roi],[0],mask=None,histSize=[500],ranges=[0,255])
        cv2.normalize(self.query_hist,self.query_hist,0,255,cv2.NORM_MINMAX)

    def find_center(self,im):
        '''actually do the tracking!'''
        if self.state == self.ROI_SELECTED:
            im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            track_im = cv2.calcBackProject([im_hsv],[0],self.query_hist,[0,255],1)

            # cv2.imshow('heatmap',track_im)

            track_im_visualize = track_im.copy()
            # convert to (x,y,w,h)
            track_roi = (self.last_detection[0],self.last_detection[1],self.last_detection[2]-self.last_detection[0],self.last_detection[3]-self.last_detection[1])
            # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            # this is done to plot intermediate results of mean shift

            #change this to use contours/connected component instead of mean shift?
            for max_iter in range(1,10):
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1 )
                (ret, intermediate_roi) = cv2.meanShift(track_im,track_roi,term_crit)
                cv2.rectangle(track_im_visualize,(intermediate_roi[0],intermediate_roi[1]),(intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]),max_iter/10.0,2)
            
            self.last_detection = [intermediate_roi[0],intermediate_roi[1],intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]]
            
            #find the average value in detection to get detection probability
            x_min,y_min,x_max,y_max = self.last_detection
            prob = (255-np.mean(track_im[x_min:x_max,y_min:y_max]))/255.0

            if prob>.97:
                self.confidence_count += 1
                if self.confidence_count >= 5:
                    self.query_img = im
                    self.query_roi = self.last_detection
                    self.get_query_histogram()
                    self.confidence_count = 0
            else:
                self.confidence_count = 0


            #get the center of the box
            posX = (self.last_detection[0]+self.last_detection[2])/2
            posY = (self.last_detection[1]+self.last_detection[3])/2

            # cv2.circle(im,(posX,posY),2,(255,0,0),10)
            # cv2.imshow('image',im)
            # cv2.rectangle(self.query_img,(self.query_roi[0],self.query_roi[1]),(self.query_roi[2],self.query_roi[3]),1.0,2)
            # cv2.imshow('query_img',self.query_img)
            # cv2.waitKey(20)

            return prob, [posX, posY, self.query_img.shape[0], self.query_img.shape[1]]
        else:
            return 0, [-1, -1, -1, -1]
