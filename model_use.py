import pickle
import random
import xml.etree.ElementTree as ET
import math
import segmentation_models as sm
import tf as tf
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import PIL
from PIL import ImageOps
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
import tensorflow as tf
from cvat_manipulator import Cvat_manipulator
import argparse
import imutils
import cv2
import numpy as np
from scipy.ndimage.measurements import label
import os
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util._montage import montage
from collections import deque
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
img_size =(800,800)



if __name__ == '__main__':

    sm.set_framework('tf.keras')
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    model = Unet(BACKBONE, input_shape=(img_size[0], img_size[1], 3), encoder_weights='imagenet', classes=2)
    model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=[iou_score])

    model.load_weights('data/weightsfile.h5')

    zero_matrix = np.zeros((1, 800, 800))
    #image = Image.open("C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_2\\images\\frame_000341.PNG")
    image =  Image.open("C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\test.png")
    image = image.resize(img_size)
    data = np.asarray(image).astype('float32')

    data = np.expand_dims(data, axis=0)
    x = model.predict(data)[0]

    mask = np.argmax(x, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    x = keras.preprocessing.image.array_to_img(mask)
    plt.imshow(x, interpolation='nearest')
    plt.show()
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))


    plt.imshow(img, interpolation='nearest')
    plt.imshow(image, interpolation='nearest', alpha=0.6)
    plt.show()



    a = mask.reshape((800,800))
    lbl_0 = label(mask)
    props = regionprops(lbl_0)
    for prop in props:
        print('Found bbox', prop.centroid)
