import pickle
import xml.etree.ElementTree as ET
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
source_width = 512
source_height = 512

if __name__ == "__main__":
    tree = ET.parse('data/dataset_1/segmentation_dataset/annotations.xml')
    photo_dir ="data/dataset_1/segmentation_dataset/images/"
    root = tree.getroot()
    mask_lst = []
    image_lst = []
    for i in root:
        if i.tag=="image":
            img = tf.keras.preprocessing.image.load_img(photo_dir + i.attrib['name'] + ".PNG",
                                                  target_size=(source_width, source_height))

            image_lst.append(tf.keras.preprocessing.image.img_to_array(img))
            width = int(i.attrib['width'])
            height = int(i.attrib['height'])
            Rx = source_width/width
            Ry = source_height/height
            mask = np.zeros((source_width,source_height),dtype=np.uint8)

            for img in i:
                if img.attrib['label'] == "needle holder":
                    needle_holder_point = img.attrib['points']
                    point = [float(k) for k in needle_holder_point.split(',')]
                    point = [Rx*point[0],Ry*point[1]]
                    round_up = [math.ceil(num) for num in point]
                    round_down = [math.floor(num) for num in point]
                    mask[round_up[0],round_up[1]] = 1
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
            mask_lst.append(mask.transpose())

    mask_lst = np.array(mask_lst)
    image_lst = np.array(image_lst)

    dataset = tf.data.Dataset.from_tensor_slices((image_lst, mask_lst))
    tf.data.experimental.save(
        dataset, "dataset.tf", compression=None, shard_func=None, checkpoint_args=None
    )
    plt.figure(dpi=1200)
    plt.imshow(image_lst[0], 'gray', interpolation='none')
    plt.imshow(mask_lst[0].transpose(), 'jet', interpolation='none', alpha=0.7, )
    plt.show()
    print("done")