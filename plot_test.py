import time

import cv2

from cvat_manipulator import Cvat_manipulator

obj = Cvat_manipulator("C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\annotations.xml","C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\images")
img,np_img = obj.get_img(500)
cv2.imshow('Tracking',np_img)
cv2.waitKey(0)
