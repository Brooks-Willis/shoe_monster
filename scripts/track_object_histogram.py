#!/usr/bin/env python

import cv2
import pickle
import numpy as np

class ObjectTracker:
	SELECTING_QUERY_IMG = 0
	SELECTING_ROI_PT_1 = 1
	SELECTING_ROI_PT_2 = 2

	""" Object Tracker shows the basics of tracking an object based on keypoints """
	def __init__(self, target_img):
		self.query_img = target_img
		self.last_detection = None

		self.state = ObjectTracker.SELECTING_QUERY_IMG

	def get_query_histogram(self):
		# set up the ROI for tracking
		roi = self.query_img.img[self.query_img.roi[1]:self.query_img.roi[3],self.query_img.roi[0]:self.query_img.roi[2],:]
		hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		# play with the number of histogram bins by changing histSize
		self.query_hist = cv2.calcHist([hsv_roi],[0],mask=None,histSize=[256],ranges=[0,255])
		cv2.normalize(self.query_hist,self.query_hist,0,255,cv2.NORM_MINMAX)

	def track(self,im):
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

		self.last_detection = [intermediate_roi[0],intermediate_roi[1],intermediate_roi[0]+intermediate_roi[2],intermediate_roi[1]+intermediate_roi[3]]
		cv2.imshow("track_win",track_im_visualize)

class TargetImg(object):
	SELECTING_ROI_PT_1 = 0
	SELECTING_ROI_PT_2 = 1
	ROI_SELECTED = 2

	def __init__(self,img):
		self.img = img
		self.roi = None
		self.state = self.SELECTING_ROI_PT_1

def mouse_event(event,x,y,flag,im):
	if event == cv2.EVENT_FLAG_LBUTTON:
		if img.state == img.SELECTING_ROI_PT_1:
			img.roi = [x,y,-1,-1]
			# cv2.circle(img.query_img_visualize,(x,y),5,(255,0,0),5)
			img.state = img.SELECTING_ROI_PT_2
		elif img.state == img.SELECTING_ROI_PT_2:
			img.roi[2:] = [x,y]
			img.last_detection = img.roi
			# cv2.circle(img.query_img_visualize,(x,y),5,(255,0,0),5)
			img.state = img.ROI_SELECTED
			tracker.get_query_histogram()
			tracker.last_detection = img.roi


if __name__ == '__main__':

	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	cv2.namedWindow("MYWIN")

	img = cv2.imread('/home/rboy/catkin_ws/src/shoe_monster/images/shoe2_right.jpg')
	scale,dim = max([(float(frame.shape[i])/img.shape[i],i) for i in [0,1]])
	img = np.array(cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale))))
	crop_dim = 1-dim
	crop_min = (img.shape[crop_dim]-frame.shape[crop_dim])/2
	crop_max = crop_min+frame.shape[crop_dim]
	if crop_dim == 1:
		img = img[:,crop_min:crop_max]
	else:
		img = img[crop_min:crop_max,:]

	img = TargetImg(img)

	cv2.namedWindow("OBJSEL")
	cv2.setMouseCallback("OBJSEL",mouse_event)
	cv2.imshow("OBJSEL",img.img)

	tracker = ObjectTracker(img)

	while True:
		ret, frame = cap.read()
		frame = np.array(cv2.resize(frame,(frame.shape[1],frame.shape[0])))

		if tracker.query_img.state == TargetImg.ROI_SELECTED:
			tracker.track(frame)

			# add the query image to the side
			combined_img = np.zeros((frame.shape[0],frame.shape[1]+(tracker.query_img.roi[2]-tracker.query_img.roi[0]),frame.shape[2]),dtype=frame.dtype)
			combined_img[:,0:frame.shape[1],:] = frame
			combined_img[0:(tracker.query_img.roi[3]-tracker.query_img.roi[1]),frame.shape[1]:,:] = (
					tracker.query_img.img[tracker.query_img.roi[1]:tracker.query_img.roi[3],
									  tracker.query_img.roi[0]:tracker.query_img.roi[2],:])

			cv2.rectangle(combined_img,(tracker.last_detection[0],tracker.last_detection[1]),(tracker.last_detection[2],tracker.last_detection[3]),(0,0,255),2)
			cv2.imshow("MYWIN",combined_img)
		else:
			cv2.imshow("MYWIN",frame)
		cv2.waitKey(50)
	cv2.destroyAllWindows()
