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

from cvat_manipulator import Cvat_manipulator
from kalman_filter import KalmanFilter

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
img_size = (1024, 1024)


def calc_euklid_distance(point_x, point_y):
    dist = math.sqrt((point_x[0] - point_y[0]) ** 2 + (point_x[1] - point_y[1]) ** 2)
    return dist


def load_model(path='data/weightsfile(1024, 1024)_resnet34_batch_1.h5'):
    sm.set_framework('tf.keras')
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    model = Unet(BACKBONE, input_shape=(img_size[0], img_size[1], 3), encoder_weights='imagenet', classes=2)
    model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=[iou_score])
    model.load_weights(path)
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
    model = load_model()

    obj = Cvat_manipulator(
        "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_2\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_2\\images",size=img_size)

    iterator = iter(obj)
    elem = next(iterator)

    for i in range(0):
        elem = next(iterator)
    lenght = elem.get_lenght()

    lst_dst = []

    for i in tqdm(range(lenght)):
        try:

            elem = next(iterator)
            point = elem.get_point()
            point = point["needle_holder"]

            img, np_img = elem.get_img()


            prediction = predict(np_img, model,plot=False)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            oper_lst = []
            # visualiation part
            if point != None:
                cv2.circle(np_img, (int(point[0]), int(point[1])), radius=5, color=(0, 0, 255), thickness=-1)
            for i in prediction:
                cv2.circle(np_img, (int(i[1]), int(i[0])), radius=5, color=(0, 255, 0), thickness=-1)
                i = (i[1],i[0])
                dst = calc_euklid_distance(i, point)
                oper_lst.append(dst)
            try:
                x = min(oper_lst)
            except:
                if point != None:
                    x = -1
                else:
                    x = 0
            lst_dst.append(x)

            cv2.imshow('Tracking', np_img)

            key = cv2.waitKey(1)
        except StopIteration:
            break
    plt.plot(lst_dst, label=f"")
    plt.title(f"Euklidovská vzdálenost od nástroje")
    plt.ylabel("Chyba")
    plt.xlabel("Snímek")
    plt.legend()
    plt.savefig(f"plot_dir/1024_1024_resnet.pdf")
    plt.show()
