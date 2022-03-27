from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from numpy import random




# keep looping
def gen_frames():
    vs = cv2.VideoCapture('tmp/cd33a266-0c0d-428c-b758-f3ba80e6d154_1.mp4')
    pts = deque(maxlen=64)  # buffer size
    color = np.random.randint(0, 255, (100, 3))
    ret, old_frame = vs.read()
    old_frame = imutils.resize(old_frame, width=1800)
    mask = np.zeros_like(old_frame)
    i = 0
    j = 330
    ct = 0
    while True:
        ret,frame = vs.read()

        if frame is None:
            break
        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=1800)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x=i
        y=j
        i+=2
        j=j-random.randint(-10,10)
        i=i+random.randint(-10,10)
        print (int(i%330))
        ct+=10

        timestamps = vs.get(cv2.CAP_PROP_POS_MSEC)
        if (timestamps):
            print (i,j, "Drawing")
            #cv2.circle(frame,(i, j),10, (0,0,255), -1) #draw circle
            mask = cv2.line(mask, (x,y),(i,j),[255,0,9], 2)
            frame = cv2.circle(frame,(i,j),5,[0,255,222],-1)

        img = cv2.add(frame,mask)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
