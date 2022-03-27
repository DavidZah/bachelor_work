import cv2
import numpy as np

from cvat_manipulator import Cvat_manipulator

obj = Cvat_manipulator("C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\annotations.xml","C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\images")

img,np_img = obj.get_img(99)



tracker = cv2.TrackerMIL_create()
video = cv2.VideoCapture('C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\video.mp4')
start_frame_number = 95
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
ok,frame=video.read()
bbox = cv2.selectROI(np_img)
ok = tracker.init(np_img,bbox)
while True:
    ok,frame=video.read()
    if not ok:
        break
    ok,bbox=tracker.update(frame)
    if ok:
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
    else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Tracking',frame)
    if cv2.waitKey(1) & 0XFF==27:
        break
cv2.destroyAllWindows()