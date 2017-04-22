# import the necessary packages
from __future__ import print_function
from scipy.spatial import distance as dist
from pyimagesearch.motion_detection import SingleMotionDetector

import numpy as np
import argparse
import imutils
import time
import datetime
import cv2

#documentation
#https://gurus.pyimagesearch.com/lessons/surveillance-and-motion-detection/

videoFile="SCV_stimpack\\2015_04_04_12_08_16_short.mp4"

MIN_FRAMES = 3 # a minimum of 1 images with a motion is needed to validate the motion
OUTPUT_DIRECTORY="SCV_stimpack"
THRESHOLD_TABLE_AREA = 5 #useqd to detect the table area
RESIZED_WIDTH = 600

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputFile= OUTPUT_DIRECTORY+"\\output.mp4"
out = cv2.VideoWriter(outputFile,fourcc, 20.0, (600, 450)) #20 fps + resolution


print("[INFO] starting video file thread...")
capture = cv2.VideoCapture(videoFile) #the cam feed is read in a separate thread

# initialize the motion detector, the total number of frames read thus far, the
# number of consecutive frames that have contained motion, and the spatial
# dimensions of the frame
md = SingleMotionDetector(accumWeight=0.1) #accumulated weight is the weight of the older frame (the larger the less)
total = 0 #is the total number of frames read from the video stream thus far
consec = None #keeps track of the number of consecutive frames that have contained motion
frameShape = None #will store the spatial dimensions of the frame read from the video stream.
cX=0
cY=0
polySelected = False
refPt = []
# loop over frames from the video file stream

def click(event, x, y, flags, param):
        #more doc http://docs.opencv.org/3.1.0/d7/dfc/group__highgui.html
        global refPt
        if event == cv2.EVENT_LBUTTONDOWN:
                refPt.append((x,y)) #add the coordinates to the polygone coordinates
        return

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click) #Anytime a mouse event happens, OpenCV will relay the pertinent details to our click function

while True:
    # grab the frame from the video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels), blur it
    (grabbed, frame) = capture.read()
    frame = imutils.resize(frame, width=RESIZED_WIDTH)
    frameCopy= frame.copy()  #will be used to take a picture of the car crossing the flash area
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # if we do not already have the dimensions of the frame, initialize it to the center of the frame
    if frameShape is None:
            frameShape = (int(gray.shape[0] / 2), int(gray.shape[1] / 2))


    # grab the current timestamp and draw it on the frame
    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, "size frame:{}/{}".format(frame.shape[0],frame.shape[1]),
            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
   
    #Catch 4 mouse clicks, create a mask with the ploygon they form.
	#source:http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    if len(refPt) == 4: #polygon is completed, can build the mask
        #http://opencvinpython.blogspot.be/2014/09/basic-drawing-examples-drawing-line-cv2.html
        polySelected = True 
        pts =np.array(refPt, np.int32)
        pts =pts.reshape((-1,1,2))
        mask=np.zeros(frame.shape[:2], dtype="uint8")
        cv2.fillConvexPoly(mask, pts, 255)
        refPt = [] #reinitialize the counter
        '''
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        '''

# if the total number of frames has reached a sufficient number to construct a
# reasonable background model and the masked image is completed, then continue to process the frame
    if total > 32 and polySelected == True:
            # detect motion in the masked image
            gray = cv2.bitwise_and(gray, gray, mask=mask) #detect moves only on the selected area
            '''
            cv2.imshow("Mask", gray)
            cv2.waitKey(0)
            '''
            motion = md.detect(gray, tVal=25, show=True)
            # if the `motion` object not None, then motion has occurred in the image
            if motion is not None:
                    # unpack the motion tuple, compute the center (x, y)-coordinates of the
                    # bounding box, and draw the bounding box of the motion on the output frame
                    (thresh, (minX, minY, maxX, maxY)) = motion
                    cX = int((minX + maxX) / 2)
                    cY = int((minY + maxY) / 2)

                    # if the number of consecutive frames is None, initialize it using a list
                    # of the number of total frames, the frame itself, along with distance of
                    # the centroid to the center of the image
                    if consec is None:
                            consec = [1, frame, dist.euclidean((cY, cX), frameShape)] #used to categorize the images

                    # otherwise, we are already detecting a move candidate and
                    # we need to update the bookkeeping variable
                    else:
                            # compute the Euclidean distance between the motion centroid and the
                            # center of the frame, then increment the total number of *consecutive frames* 
                            # that contain motion
                            d = dist.euclidean((cY, cX), frameShape)
                            consec[0] += 1

                            # if the distance is smaller than the current distance, then update the
                            # bookkeeping variable
                            if d < consec[2]:
                                    consec[1:] = (frame, d)

                    # if a sufficient number of frames have contained motion, log the motion
                    if consec[0] > MIN_FRAMES:
                        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0,255,0), 3)
                        out.write(frame)
                        consec = None
            # otherwise, there is no motion in the frame so reset the consecutive bookkeeping
            # variable
            else:
                consec = None
    # update the background model and increment the total number of frames read thus far
    md.update(gray)
    total += 1

# show the frame and record if the user presses a key
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
            break

# clean up the camera and close any open windows
cv2.destroyAllWindows()
out.release()

