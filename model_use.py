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


img_size =(800,800)


if __name__ == '__main__':

    sm.set_framework('tf.keras')
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    model = Unet(BACKBONE, input_shape=(img_size[0], img_size[1], 3), encoder_weights='imagenet', classes=2)
    model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=[iou_score])

    model.load_weights('data/weightsfile.h5')

    zero_matrix = np.zeros((1, 800, 800))
    image = Image.open("C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_2\\images\\frame_000253.PNG")
    image = image.resize(img_size)
    data = np.asarray(image).astype('float32')

    data = np.expand_dims(data, axis=0)
    x = model.predict(data)[0]

    mask = np.argmax(x, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))

    plt.imshow(img, interpolation='nearest')
    plt.imshow(image, interpolation='nearest', alpha=0.6)
    plt.show()