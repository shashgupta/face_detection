import argparse
import imutils
import cv2
import os
import numpy as np

FACE_DETECTOR_PATH =  "./haarcascade_frontalface_default.xml"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to image file")
args = vars(ap.parse_args())


img = cv2.imread(args["image"])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize = (30,30), flags=cv2.CASCADE_SCALE_IMAGE)

rects = [(int(x), int(y), int(x+w), int(y+h)) for (x,y,w,h) in rects]
#print(rects)

for(i, c) in enumerate(rects):
    cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (0,255,0), 2)
    cv2.putText(img, "Object #{}".format(i+1), (int(c[0] - 15), int(c[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
	
cv2.imshow("face", img)

cv2.waitKey(0)
