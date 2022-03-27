import pickle
import xml.etree.ElementTree as ET
import math
import segmentation_models as sm
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

input_dir = "data/dataset_1/segmentation_dataset/images"
target_dir = "masks/"
sm.set_framework('tf.keras')
img_size = (512,512)
batch_size = 4

class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = batch_input_img_paths[j]

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            z = np.expand_dims(batch_target_img_paths[j], 2)
            y[j] = z
        return x, y

def get_mask(id):
    tree = ET.parse('data/dataset_1/segmentation_dataset/annotations.xml')
    root = tree.getroot()
    mask_lst = []
    for i in root:
        if i.tag == "image":
            width = int(i.attrib['width'])
            height = int(i.attrib['height'])
            mask = np.zeros((width, height))
            for img in i:
                if img.attrib['label'] == "needle holder":
                    needle_holder_point = img.attrib['points']
                    point = [float(k) for k in needle_holder_point.split(',')]
                    round_up = [math.ceil(num) for num in point]
                    round_down = [math.floor(num) for num in point]
                    mask[round_up[0], round_up[1]] = 1
                    mask[round_up[0], round_down[1]] = 1
                    mask[round_down[0], round_up[1]] = 1
                    mask[round_down[0], round_down[1]] = 1

                if img.attrib['label'] == "scissors":
                    scissors_point = img.attrib['points']
                    point = [float(k) for k in scissors_point.split(',')]
                    round_up = [math.ceil(num) for num in point]
                    round_down = [math.floor(num) for num in point]
                    mask[round_up[0], round_up[1]] = 2
                    mask[round_up[0], round_down[1]] = 2
                    mask[round_down[0], round_up[1]] = 2
                    mask[round_down[0], round_down[1]] = 2

                if img.attrib['label'] == "tweezers":
                    tweezers = img.attrib['points']
                    point = [float(k) for k in tweezers.split(',')]
                    round_up = [math.ceil(num) for num in point]
                    round_down = [math.floor(num) for num in point]
                    mask[round_up[0], round_up[1]] = 3
                    mask[round_up[0], round_down[1]] = 3
                    mask[round_down[0], round_up[1]] = 3
                    mask[round_down[0], round_down[1]] = 3

            mask_lst.append(mask)
def load_data(split):

    with open("dataset.pck", "rb") as fp:   # Unpickling
        target_img_paths = pickle.load(fp)
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".PNG")
        ]
    )
    x_oper = []
    y_oper = []
    for index,img_path in enumerate(input_img_paths):
        img = np.array(load_img(img_path, target_size=img_size))
        x_oper.append(img)
        target = np.load(target_img_paths[index])
        y_oper.append(target)
    print(len(input_img_paths))

    x_val = x_oper[-split:]
    y_val = y_oper[-split:]

    x_train =x_oper[0:(len(x_oper)-split)]
    y_train = y_oper[0:(len(y_oper)-split)]
    return x_train, y_train, x_val, y_val




BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# load your data
x_train, y_train, x_val, y_val = load_data(10)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)


train_gen = OxfordPets(
    batch_size, img_size, x_train, y_train
)

# define model
model = Unet(BACKBONE, input_shape = (512,512,3),encoder_weights='imagenet',classes=5)
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
model.fit(train_gen,
    epochs=150
)


val_gen = OxfordPets(batch_size, img_size, x_train, y_train)
val_preds = model.predict(val_gen)

def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    plt.imshow(img)
    plt.show()

i = 2

display_mask(i)