#!/usr/bin/env python

import cv2
import numpy as np

class AbstractTracker(object):
    #States for selecting the region of interest
    SELECTING_ROI = 0
    ROI_SELECTED = 1

    def __init__(self):
        self.query_img = None
        self.query_roi = None
        self.last_detection = None
        self.state = self.SELECTING_ROI

    def set_query(self,img,roi):
        '''ROI should be [x_topleft,y_topleft,x_bottomright,y_bottomright]'''
        self.query_img = img
        self.query_roi = roi
        self.update_criteria()
        self.last_detection = roi
        self.state = self.ROI_SELECTED

    def update_criteria(self):
        '''do whatever updates you need (histogram, keypoints, etc)'''
        pass

    def track(self,im,viz=False):
        '''actually do the tracking!
        return x_center, y_center'''
        return 0, [-1, -1]


class HistTracker(AbstractTracker):
    #States for selecting the region of interest
    SELECTING_ROI = 0
    ROI_SELECTED = 1

    def __init__(self):
        super(HistTracker,self).__init__()
        self.confidence_count = 0

    def update_criteria(self):
        '''calculate the histogram to try to match'''
        # set up the ROI for tracking
        roi = self.query_img[self.query_roi[1]:self.query_roi[3],self.query_roi[0]:self.query_roi[2],:]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # play with the number of histogram bins by changing histSize
        self.query_hist = cv2.calcHist([hsv_roi],[0],mask=None,histSize=[500],ranges=[0,255])
        cv2.normalize(self.query_hist,self.query_hist,0,255,cv2.NORM_MINMAX)

    def viz(self,im,posX,posY):
        cv2.circle(im,(posX,posY),2,(255,0,0),10)
        cv2.imshow('hist_img',im)
        cv2.rectangle(self.query_img,(self.query_roi[0],self.query_roi[1]),(self.query_roi[0]+self.query_roi[2],self.query_roi[1]+self.query_roi[3]),1.0,2)
        cv2.imshow('hist_query_img',self.query_img)
        cv2.waitKey(20)

    def find_center(self,im):
        '''actually do the tracking!'''
        im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        track_im = cv2.calcBackProject([im_hsv],[0],self.query_hist,[0,255],1)

        # cv2.imshow('heatmap',track_im)
        track_im_visualize = track_im.copy()

        # convert to (x,y,w,h)
        track_roi = (self.last_detection[0],
                     self.last_detection[1],
                     self.last_detection[2]-self.last_detection[0],
                     self.last_detection[3]-self.last_detection[1])
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        # this is done to plot intermediate results of mean shift

        # TODO: change this to use contours/connected component instead of mean shift?
        for max_iter in range(1,10):
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1 )
            (ret, intermediate_roi) = cv2.meanShift(track_im,track_roi,term_crit)
            cv2.rectangle(track_im_visualize,(intermediate_roi[0],intermediate_roi[1]),(intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]),max_iter/10.0,2)
        
        self.last_detection = [intermediate_roi[0],intermediate_roi[1],intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]]
        #find the average value in detection to get detection probability
        x_min,y_min,x_max,y_max = self.last_detection
        self.prob = (255-np.mean(track_im[x_min:x_max,y_min:y_max]))/255.0

        #if we're really confident, update the query image
        if self.prob>.97:
            self.confidence_count += 1
            if self.confidence_count >= 5:
                self.set_query(im,self.last_detection)
                self.confidence_count = 0
        else:
            self.confidence_count = 0

    def track(self,im,viz=False):
        '''actually do the tracking!'''
        if self.state == self.ROI_SELECTED:

            self.find_center(im)

            #get the center of the box
            posX = (self.last_detection[0]+self.last_detection[2])/2
            posY = (self.last_detection[1]+self.last_detection[3])/2
    
            if viz:
                self.viz(im,posX,posY)

            return self.prob, [posX, posY]
        else:
            return 0, [-1, -1]

class KeypointTracker(AbstractTracker):
    #States for selecting the region of interest
    SELECTING_ROI = 0
    ROI_SELECTED = 1

    def __init__(self, descriptor_name='SURF'):
        super(KeypointTracker,self).__init__()

        self.detector = cv2.FeatureDetector_create(descriptor_name)
        self.extractor = cv2.DescriptorExtractor_create(descriptor_name)
        self.matcher = cv2.BFMatcher()

        self.corner_threshold = 0.0
        self.ratio_threshold = 1.0

    def update_criteria(self):
        query_img_bw = cv2.cvtColor(self.query_img,cv2.COLOR_BGR2GRAY)
        kp = self.detector.detect(query_img_bw)
        kp = [pt
              for pt in kp if (pt.response > self.corner_threshold and
                               self.query_roi[0] <= pt.pt[0] < self.query_roi[2] and
                               self.query_roi[1] <= pt.pt[1] < self.query_roi[3])]
        dc, des = self.extractor.compute(query_img_bw,kp)
        # remap keypoints so they are relative to the query ROI
        for pt in kp:
            pt.pt = (pt.pt[0] - self.query_roi[0], pt.pt[1] - self.query_roi[1])
        self.query_keypoints = kp
        self.query_descriptors = des

    def viz(self,im):
        # add the query image to the side
        combined_img = np.zeros((im.shape[0],im.shape[1]+(self.query_roi[2]-self.query_roi[0]),im.shape[2]),dtype=im.dtype)
        combined_img[:,0:im.shape[1],:] = im
        combined_img[0:(self.query_roi[3]-self.query_roi[1]),im.shape[1]:,:] = (
                self.query_img[self.query_roi[1]:self.query_roi[3],
                                  self.query_roi[0]:self.query_roi[2],:])
        # plot the matching points and correspondences
        for i in range(self.matching_query_pts.shape[0]):
            cv2.circle(combined_img,(int(self.matching_training_pts[i,0]),int(self.matching_training_pts[i,1])),2,(255,0,0),2)
            cv2.line(combined_img,(int(self.matching_training_pts[i,0]), int(self.matching_training_pts[i,1])),
                                  (int(self.matching_query_pts[i,0]+im.shape[1]),int(self.matching_query_pts[i,1])),
                                  (0,255,0))

        for pt in self.query_keypoints:
            cv2.circle(combined_img,(int(pt.pt[0]+im.shape[1]),int(pt.pt[1])),2,(255,0,0),1)
        cv2.rectangle(combined_img,(self.last_detection[0],self.last_detection[1]),(self.last_detection[2],self.last_detection[3]),(0,0,255),2)

        cv2.imshow("keypoint_img",combined_img)

    def find_center(self,im):
        '''actually do the tracking!'''
        im_bw = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        training_keypoints = self.detector.detect(im_bw)
        #print len(training_keypoints)
        dc, training_descriptors = self.extractor.compute(im_bw,training_keypoints)

        matches = self.matcher.knnMatch(self.query_descriptors,training_descriptors,k=2)
        good_matches = []
        for m,n in matches:
            # make sure the distance to the closest match is sufficiently better than the second closest
            if (m.distance < self.ratio_threshold*n.distance and
                training_keypoints[m.trainIdx].response > self.corner_threshold):
                good_matches.append((m.queryIdx, m.trainIdx))

        self.matching_query_pts = np.zeros((len(good_matches),2))
        self.matching_training_pts = np.zeros((len(good_matches),2))
        track_im = np.zeros(im_bw.shape)
        for idx in range(len(good_matches)):
            match = good_matches[idx]
            self.matching_query_pts[idx,:] = self.query_keypoints[match[0]].pt
            self.matching_training_pts[idx,:] = training_keypoints[match[1]].pt
            track_im[training_keypoints[match[1]].pt[1],training_keypoints[match[1]].pt[0]] = 1.0

        track_im_visualize = track_im.copy()

        # convert to (x,y,w,h)
        track_roi = (self.last_detection[0],self.last_detection[1],self.last_detection[2]-self.last_detection[0],self.last_detection[3]-self.last_detection[1])
        
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        # this is done to plot intermediate results of mean shift
        for max_iter in range(1,10):
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1 )
            (ret, intermediate_roi) = cv2.meanShift(track_im,track_roi,term_crit)
            cv2.rectangle(track_im_visualize,(intermediate_roi[0],intermediate_roi[1]),(intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]),max_iter/10.0,2)

        self.last_detection = [intermediate_roi[0],intermediate_roi[1],intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]]
   

    def track(self,im,viz=False):
        '''actually do the tracking!'''
        if self.state == self.ROI_SELECTED:

            self.find_center(im)

            #get the center of the box
            posX = (self.last_detection[0]+self.last_detection[2])/2
            posY = (self.last_detection[1]+self.last_detection[3])/2

            if viz:
                self.viz(im)
            return .71, [posX, posY]
        else:
            return 0, [-1, -1]
