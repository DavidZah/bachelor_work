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

from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
import tensorflow as tf
from cvat_manipulator import Cvat_manipulator
sm.set_framework('tf.keras')
BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)
model = Unet(BACKBONE, input_shape=(512, 512, 3), encoder_weights='imagenet', classes=2)

model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=[iou_score])

model.load_weights('data/model/')

