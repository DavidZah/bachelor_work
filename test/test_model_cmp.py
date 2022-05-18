import math

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
import PIL
from PIL import ImageOps
from tensorflow import keras
from scipy.ndimage.measurements import label
import os
import cv2
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm

from kalman_filter import KalmanFilter

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
img_size = (800, 800)


def load_model(path='data/weightsfile.h5'):
    sm.set_framework('tf.keras')
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    model = Unet(BACKBONE, input_shape=(img_size[0], img_size[1], 3), encoder_weights='imagenet', classes=2)
    model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=[iou_score])
    model.load_weights('data/weightsfile.h5')
    return model


def predict(frame, model, plot=False):
    ret = []
    data = np.asarray(frame).astype('float32')
    data = np.expand_dims(data, axis=0)

    x = model.predict(data)[0]

    mask = np.argmax(x, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    if plot:
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        plt.imshow(img, interpolation='nearest')
        plt.imshow(frame, interpolation='nearest', alpha=0.6)
        plt.show()

    lbl_0 = label(mask)
    props = regionprops(lbl_0)
    for prop in props:
        ret.append(prop.centroid)
    return ret


if __name__ == '__main__':
    buffer = 128
    model = load_model()
    VideoCap = cv2.VideoCapture('data\\video_test_3.mov')
    length = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))

    HiSpeed = 100
    ControlSpeedVar = 30  # Lowest: 1 - Highest:100
    debugMode = 1

    points_lst = []
    for i in tqdm(range(length)):
        try:
            ret, frame = VideoCap.read()
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            points_lst.append(predict(frame,model))
        except:
            break

    print("bleh")
    while (True):
        # Read frame
        ret, frame = VideoCap.read()
        frame = cv2.resize(frame, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect object
        centers = predict(frame, model, plot=False)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for i in centers:
            cv2.circle(frame, (int(i[1]), int(i[0])), radius=5, color=(0, 0, 255), thickness=-1)

        cv2.imshow('Tracking', frame)
        key = cv2.waitKey(10)