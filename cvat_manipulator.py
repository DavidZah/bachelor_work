import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math

class Point:
    def __init__(self,label,x,y):
        self.label = label
        self.x = x
        self.y = y
    def get_position(self):
        return [self.x,self.y]

class Cvat_manipulator:
    #todo fix auto size
    def __init__(self,xml_path,photo_dir,size = (1366,768),img_extension = ".PNG"):
        self.xml_path = xml_path
        self.photo_dir = photo_dir
        self.photo = []
        self.load_xml()
        self.img_extension = img_extension
        self.size = size
        self.index = 0
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index < len(self.photo)-1:
            self.index +=1
            return self
        else:
            raise StopIteration

    def get_lenght(self):
        left = len(self.photo)-self.index
        return left
    def load_xml(self):
        try:
            self.xml = ET.parse(self.xml_path)
        except:
            raise  FileNotFoundError

        root = self.xml.getroot()
        for i in root:
            if i.tag == "image":
                self.photo.append(i)

    def get_img(self,idx = None):
        if (idx == None):
            idx = self.index
        img = Image.open(Path(self.photo_dir,self.photo[idx].attrib['name']+self.img_extension))
        img_npy = np.array(img)
        #img_npy = img_npy[:,:,::-1]
        return img,img_npy

    def get_point(self,idx = None,labels = None):
        if (idx == None):
            idx = self.index
        i = self.photo[idx]
        width = int(i.attrib['width'])
        height = int(i.attrib['height'])
        Rx = self.size[0] / width
        Ry = self.size[1] / height
        #todo need to done more universali
        point_needle_holder = None
        point_scissors = None
        point_tweezers = None
        for img in i:
            if img.attrib['label'] == "needle holder":
                needle_holder_point = img.attrib['points']
                point = [float(k) for k in needle_holder_point.split(',')]
                point_needle_holder = [Rx * point[0], Ry * point[1]]


            if img.attrib['label'] == "scissors":
                scissors_point = img.attrib['points']
                point = [float(k) for k in scissors_point.split(',')]
                point_scissors = [Rx * point[0], Ry * point[1]]


            if img.attrib['label'] == "tweezers":
                tweezers = img.attrib['points']
                point = [float(k) for k in tweezers.split(',')]
                point_tweezers = [Rx * point[0], Ry * point[1]]
        return {
            "needle_holder" : point_needle_holder,
            "scissors":point_scissors,
            "tweezers":point_tweezers
            }

    def get_mask(self,idx = None):
        if (idx == None):
            idx = self.index
        i = self.photo[idx]
        width = int(i.attrib['width'])
        height = int(i.attrib['height'])
        Rx = self.size[0] / width
        Ry = self.size[1] / height
        mask = np.zeros((self.size[0], self.size[1]), dtype=np.uint8)
        for img in i:
            if img.attrib['label'] == "needle holder":
                needle_holder_point = img.attrib['points']
                point = [float(k) for k in needle_holder_point.split(',')]
                point = [Rx * point[0], Ry * point[1]]
                round_up = [math.ceil(num) for num in point]
                round_down = [math.floor(num) for num in point]
                mask[round_up[0], round_up[1]] = 1
                mask[round_up[0], round_down[1]] = 1
                mask[round_down[0], round_up[1]] = 1
                mask[round_down[0], round_down[1]] = 1

            if img.attrib['label'] == "scissors":
                scissors_point = img.attrib['points']
                point = [float(k) for k in scissors_point.split(',')]
                point = [Rx * point[0], Ry * point[1]]
                round_up = [math.ceil(num) for num in point]
                round_down = [math.floor(num) for num in point]
                mask[round_up[0], round_up[1]] = 2
                mask[round_up[0], round_down[1]] = 2
                mask[round_down[0], round_up[1]] = 2
                mask[round_down[0], round_down[1]] = 2

            if img.attrib['label'] == "tweezers":
                tweezers = img.attrib['points']
                point = [float(k) for k in tweezers.split(',')]
                point = [Rx * point[0], Ry * point[1]]
                round_up = [math.ceil(num) for num in point]
                round_down = [math.floor(num) for num in point]
                mask[round_up[0], round_up[1]] = 3
                mask[round_up[0], round_down[1]] = 3
                mask[round_down[0], round_up[1]] = 3
                mask[round_down[0], round_down[1]] = 3
        return mask.transpose()

if __name__ == '__main__':
    obj = Cvat_manipulator("data/dataset_1/segmentation_dataset/annotations.xml","data/dataset_1/segmentation_dataset/images")
    iterator = iter(obj)
    elem = next(iterator)
    print(elem.get_point())
    plt.show()
    print("done")