import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cvat_manipulator import Cvat_manipulator

def calc_euklid_distance(point_x,point_y):
    dist = math.sqrt((point_x[0] - point_y[0]) ** 2 + (point_x[1] - point_y[1]) ** 2)
    return dist


def box_to_point(bbox):
    (x, y, w, h) = [int(v) for v in bbox]
    x = x - w/2
    y = y-h/2
    return [x,y]

def point_to_box(point,size = 10):
    y = math.sqrt(size/2)
    tup = (round(point[0]-y),round(point[1]-y),surroundigs,surroundigs)
    return tup

if __name__ == '__main__':

        lst_dst = []
        obj = Cvat_manipulator("C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\annotations.xml","C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\images")
        tracker = cv2.TrackerMIL_create()
        iterator = iter(obj)
        for i in range(0,500):
            elem = next(iterator)
        point = elem.get_point()
        point = point["needle_holder"]
        surroundigs = 20
        img,np_img = elem.get_img()
        cv2.imshow('Tracking', np_img)
        bbox = point_to_box(point,surroundigs)
        #visual_bbox = cv2.selectROI(np_img)
        #ok = tracker.init(np_img, (400,40,425,425))
        ok = tracker.init(np_img, bbox)
        lenght = elem.get_lenght()
        for i in tqdm(range(lenght)):
            try:
                # Get next element from TeamIterator object using iterator object
                elem = next(iterator)
                point = elem.get_point()
                point = point["needle_holder"]
                if point != None:

                    img,np_img = elem.get_img()
                    ok, bbox = tracker.update(np_img)
                    (x, y, w, h) = [int(v) for v in bbox]



                    #visualiation part
                    cv2.circle(np_img, (int(point[0]), int(point[1])), radius=1, color=(0, 0, 255), thickness=-1)
                    cv2.rectangle(np_img, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)

                    bbox_point = box_to_point(bbox)
                    dst = calc_euklid_distance(bbox_point,point)
                    lst_dst.append(dst)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(np_img,str(int(dst)), (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow('Tracking', np_img)

                    key = cv2.waitKey(30)
            except StopIteration:
                break
        plt.plot(lst_dst)
        plt.title(str(surroundigs))
        plt.show()
        with open(f"{str(surroundigs)}.txt", "w") as output:
            output.write(str(lst_dst))
print("done")