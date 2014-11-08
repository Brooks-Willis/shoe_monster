#!/usr/bin/env python

import cv2
import pickle
import numpy as np


class ObjectTracker:
	SELECTING_QUERY_IMG = 0
	SELECTING_ROI_PT_1 = 1
	SELECTING_ROI_PT_2 = 2

	""" Object Tracker shows the basics of tracking an object based on keypoints """
	def __init__(self, descriptor_name):
		self.detector = cv2.FeatureDetector_create(descriptor_name)
		self.extractor = cv2.DescriptorExtractor_create(descriptor_name)
		self.matcher = cv2.BFMatcher()
		self.query_img = None
		self.query_roi = None
		self.query_img_test = None #For testing update_query_roi
		self.query_roi_test = None #For testing update_query_roi
		self.last_detection = None

		self.corner_threshold = 0.0
		self.ratio_threshold = 1.0

		self.state = ObjectTracker.SELECTING_QUERY_IMG

	def set_ratio_threshold(self,thresh):
		self.ratio_threshold = thresh

	def set_corner_threshold(self,thresh):
		self.corner_threshold = thresh

	def get_query_keypoints(self):
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

	def update_query_roi(self, im, new_roi):
		"""Takes image and vector of 
		[upper left x, upper left y, lower right x, lower right y]"""
		self.query_img = im
		self.query_roi = new_roi
		self.get_query_keypoints

	def track(self,im):
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
		print good_matches
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

		cv2.imshow("track_win",track_im_visualize)

def set_corner_threshold_callback(thresh):
	""" Sets the threshold to consider an interest point a corner.  The higher the value
		the more the point must look like a corner to be considered """
	tracker.set_corner_threshold(thresh/1000.0)

def set_ratio_threshold_callback(ratio):
	""" Sets the ratio of the nearest to the second nearest neighbor to consider the match a good one """
	tracker.set_ratio_threshold(ratio/100.0)

def mouse_event(event,x,y,flag,im):
	if event == cv2.EVENT_FLAG_LBUTTON:
		if tracker.state == tracker.SELECTING_QUERY_IMG:
			tracker.query_img_visualize = frame.copy()
			tracker.query_img = frame
			tracker.query_roi = None
			tracker.state = tracker.SELECTING_ROI_PT_1
		elif tracker.state == tracker.SELECTING_ROI_PT_1:
			tracker.query_roi = [x,y,-1,-1]
			cv2.circle(tracker.query_img_visualize,(x,y),5,(255,0,0),5)
			tracker.state = tracker.SELECTING_ROI_PT_2
		else:
			tracker.query_roi[2:] = [x,y]
			tracker.last_detection = tracker.query_roi
			cv2.circle(tracker.query_img_visualize,(x,y),5,(255,0,0),5)
			tracker.state = tracker.SELECTING_QUERY_IMG
			tracker.get_query_keypoints()

	if event == cv2.EVENT_FLAG_RBUTTON: #This is just for testing the update_query_roi function
		if tracker.state == tracker.SELECTING_QUERY_IMG:
			tracker.query_img_visualize = frame.copy()
			tracker.query_img_test = frame
			tracker.query_roi_test = None
			tracker.state = tracker.SELECTING_ROI_PT_1
		elif tracker.state == tracker.SELECTING_ROI_PT_1:
			tracker.query_roi_test = [x,y,-1,-1]
			cv2.circle(tracker.query_img_visualize,(x,y),5,(255,0,0),5)
			tracker.state = tracker.SELECTING_ROI_PT_2
		else:
			tracker.query_roi_test[2:] = [x,y]
			tracker.last_detection = tracker.query_roi_test
			cv2.circle(tracker.query_img_visualize,(x,y),5,(255,0,0),5)
			tracker.state = tracker.SELECTING_QUERY_IMG
			tracker.update_query_roi(tracker.query_img_test,tracker.query_roi_test)


if __name__ == '__main__':
	# descriptor can be: SIFT, SURF, BRIEF, BRISK, ORB, FREAK
	tracker = ObjectTracker('SURF')

	cap = cv2.VideoCapture(0)

	cv2.namedWindow('UI')
	cv2.createTrackbar('Corner Threshold', 'UI', 0, 100, set_corner_threshold_callback)
	cv2.createTrackbar('Ratio Threshold', 'UI', 100, 100, set_ratio_threshold_callback)

	cv2.namedWindow("Chase the Shoe")
	cv2.setMouseCallback("Chase the Shoe",mouse_event)

	while True:
		ret, frame = cap.read()
		frame = np.array(cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2)))

		if tracker.state == tracker.SELECTING_QUERY_IMG:
			if tracker.query_roi != None: #Runs after selecting inital roi
				#print tracker.query_roi
				tracker.track(frame)

				# add the query image to the side
				combined_img = np.zeros((frame.shape[0],frame.shape[1]+(tracker.query_roi[2]-tracker.query_roi[0]),frame.shape[2]),dtype=frame.dtype)
				combined_img[:,0:frame.shape[1],:] = frame
				combined_img[0:(tracker.query_roi[3]-tracker.query_roi[1]),frame.shape[1]:,:] = (
						tracker.query_img[tracker.query_roi[1]:tracker.query_roi[3],
										  tracker.query_roi[0]:tracker.query_roi[2],:])
				# plot the matching points and correspondences
				for i in range(tracker.matching_query_pts.shape[0]):
					cv2.circle(combined_img,(int(tracker.matching_training_pts[i,0]),int(tracker.matching_training_pts[i,1])),2,(255,0,0),2)
					cv2.line(combined_img,(int(tracker.matching_training_pts[i,0]), int(tracker.matching_training_pts[i,1])),
										  (int(tracker.matching_query_pts[i,0]+frame.shape[1]),int(tracker.matching_query_pts[i,1])),
										  (0,255,0))

				for pt in tracker.query_keypoints:
					cv2.circle(combined_img,(int(pt.pt[0]+frame.shape[1]),int(pt.pt[1])),2,(255,0,0),1)
				cv2.rectangle(combined_img,(tracker.last_detection[0],tracker.last_detection[1]),(tracker.last_detection[2],tracker.last_detection[3]),(0,0,255),2)

				cv2.imshow("Chase the Shoe",combined_img)
			else:
				cv2.imshow("Chase the Shoe",frame)
		else:
			cv2.imshow("Chase the Shoe",tracker.query_img_visualize)
		cv2.waitKey(50)
	cv2.destroyAllWindows()
