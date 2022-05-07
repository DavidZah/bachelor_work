import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from random import random

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math


class Point:
    def __init__(self, label, x, y):
        self.label = label
        self.x = x
        self.y = y

    def get_position(self):
        return [self.x, self.y]


def replace_submatrix(mat, ind1, ind2, surroundings=5):
    ind1 = round(ind1)
    ind2 = round(ind2)

    try:
        mat[ind1 - surroundings:ind1 + surroundings, ind2 - surroundings:ind2 + surroundings] = 1
    except:
        # TODO not good solution
        for i in range(surroundings):
            try:
                mat[ind1 - surroundings - i:ind1 + surroundings - i, ind2 - surroundings:ind2 + surroundings - i] = 1
            except:
                _ = 1
    return mat


class Cvat_manipulator:
    # todo fix auto size
    def __init__(self, xml_path, photo_dir, size=(1366, 768), img_extension=".PNG", shuffle=False):
        self.xml_paths = [xml_path]
        self.photo_dirs = [photo_dir]
        self.photo = []
        self.load_xml()
        self.img_extension = img_extension
        self.size = size
        self.index = 0
        if shuffle:
            self.shuffle()

    def add_dataset(self, xml_path, photo_dir,shuffle = False):
        self.xml_paths.append(xml_path)
        self.photo_dirs.append(photo_dir)
        self.load_xml()
        if shuffle:
            self.shuffle()
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.photo) - 1:
            self.index += 1
            return self
        else:
            raise StopIteration

    def split(self, val):
        training = self.photo[:int(len(self.photo) * (1 - val))]
        validation = self.photo[-int(len(self.photo) * val):]

        train_split = copy.deepcopy(self)
        val_split = copy.deepcopy(self)
        train_split.photo = training
        val_split.photo = validation
        return train_split, val_split

    def get_lenght(self):
        left = len(self.photo) - self.index
        return left

    def load_xml(self):
        self.photo = []
        for index, i in enumerate(self.xml_paths):
            try:
                xml = ET.parse(i)
            except:
                raise FileNotFoundError

            root = xml.getroot()
            for i in root:
                if i.tag == "image":
                    i.attrib['img_path'] = self.photo_dirs[index]
                    self.photo.append(i)

    def get_img(self, idx=None):
        if (idx == None):
            idx = self.index
        img = Image.open(Path(self.photo[idx].attrib['img_path'], self.photo[idx].attrib['name'] + self.img_extension))
        img = img.resize(self.size)
        img_npy = np.array(img)
        # img_npy = img_npy[:,:,::-1]
        return img, img_npy

    def get_point(self, idx=None, labels=None):
        if (idx == None):
            idx = self.index
        i = self.photo[idx]
        width = int(i.attrib['width'])
        height = int(i.attrib['height'])
        Rx = self.size[0] / width
        Ry = self.size[1] / height

        # todo need to done more universali
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
            "needle_holder": point_needle_holder,
            "scissors": point_scissors,
            "tweezers": point_tweezers
        }

    def shuffle(self):
        np.random.shuffle(self.photo)

    def get_mask(self, idx=None, surroundings=5):
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
                mask = replace_submatrix(mask, point[0], point[1], surroundings=surroundings)

            if img.attrib['label'] == "scissors":
                scissors_point = img.attrib['points']
                point = [float(k) for k in scissors_point.split(',')]
                point = [Rx * point[0], Ry * point[1]]
            # mask = replace_submatrix(mask,point[0],point[1],surroundings=surroundings)

            if img.attrib['label'] == "tweezers":
                tweezers = img.attrib['points']
                point = [float(k) for k in tweezers.split(',')]
                point = [Rx * point[0], Ry * point[1]]
                # mask = replace_submatrix(mask,point[0],point[1],surroundings=surroundings)
        return mask.transpose()


if __name__ == '__main__':
    obj = Cvat_manipulator("data/dataset_1/segmentation_dataset/annotations.xml",
                           "data/dataset_1/segmentation_dataset/images",
                           size=(512, 512))
    obj.add_dataset(
        "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_2\\annotations.xml",
        "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_2\\images")
    print(obj.get_lenght())
    mask = obj.get_mask(4200)
    _, np_img = obj.get_img(4200)
    plt.imshow(mask, interpolation='nearest')
    plt.imshow(np_img, interpolation='nearest', alpha=0.7)
    plt.show()
    print("done")
