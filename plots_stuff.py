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



networks = ["Resnet34\n Rozlišení 512 x 512\n,BatchSize = 8",
            "Resnet34\n Rozlišení 800 x 800\n,BatchSize = 2",
            "Resnet101\n Rozlišení 800 x 800\n,BatchSize = 2",
            "Resnet34\n Rozlišení 1024 x 1024\n,BatchSize = 1"
            ]


lst = [
np.load("data/plots/lst_(512, 512)_resnet34_batch_8.npy"),
np.load("data/plots/lst_(800, 800)_resnet34_batch_2.npy"),
np.load("data/plots/lst_(800, 800)_resnet101_batch_1.npy"),
np.load("data/plots/lst_(1024, 1024)_resnet34_batch_1.npy")
]

not_fount_points = []

for i in lst:
    indexes = np.where(np.array(i) == -1)[0]
    print(len(indexes))
    not_fount_points.append(len(indexes))

fig = plt.figure(figsize = (8, 8))
plt.title("Počet nedetekovaných snímků",fontsize=20)
plt.xlabel("Testované modely",fontsize=20)
plt.ylabel("Počet snímků",fontsize=20)
plt.bar(networks, not_fount_points, color='deepskyblue',
            width=0.4)
plt.savefig(f"plot_dir/vynechane_snimky.pdf")
plt.show()

