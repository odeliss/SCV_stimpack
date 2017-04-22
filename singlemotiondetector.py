# import the necessary packages
import numpy as np
import cv2

class SingleMotionDetector:
	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		self.accumWeight = accumWeight

		# initialize the background model
		self.bg = None

	def update(self, image):
		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25, show=False):
		# compute the absolute difference between the background model and the image
		# passed in, then threshold the delta image
		# if flag show is set to true, the different intermediate steps will be shown on screen


		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
		
		# perform a series of erosions and dilations to remove small blobs
		thresh = cv2.erode(thresh, None, iterations=4)
		thresh = cv2.dilate(thresh, None, iterations=4)
		
		#show the intermediate result if required
		if show:
			cv2.imshow("Threshold", thresh)
			key = cv2.waitKey(1) & 0xFF
		
		# find contours in the thresholded image and initialize the minimum and maximum
		# bounding box regions for motion
		(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)


		# if no contours were found, return None
		if len(cnts) == 0:
			return None

        
		#Otherwize, compute the hull contours and render them on screen if required
		# loop over the contours
		
		for c in cnts:
			# compute the bounding box of the contour and use it to update the minimum
			# and maximum bounding box regions
			#hull=cv2.convexHull(c)
			#cv2.drawContours(thresh, [hull], -1, 255, -1)
			'''
			#show the intermediate result if required
			if show:
				cv2.imshow("Hull Contours", thresh)
				key = cv2.waitKey(1) & 0xFF
			'''
			(x, y, w, h) = cv2.boundingRect(c)
			
			#return the coordinates of the biggest bounding rectangle
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

		#otherwise, return a tuple of the thresholded image along with bounding box

		return (thresh, (minX, minY, maxX, maxY))
