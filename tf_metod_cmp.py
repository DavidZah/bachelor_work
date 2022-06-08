import math

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import segmentation_models as sm
from PIL import ImageOps
from scipy.ndimage.measurements import label
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
from skimage.measure import label, regionprops
from tensorflow import keras
from tqdm import tqdm

from cvat_manipulator import Cvat_manipulator

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
img_size = (1024, 1024)


def calc_euklid_distance(point_x, point_y):
    dist = math.sqrt((point_x[0] - point_y[0]) ** 2 + (point_x[1] - point_y[1]) ** 2)
    return dist


def load_model(path):
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

def plot_trajectory(lst):
    plt_org_x_lst = []
    plt_org_y_lst = []
    plt_pred_x_lst = []
    plt_pred_y_lst = []
    for i in lst:
        plt_org_x_lst.append(i[0][0])
        plt_org_y_lst.append(i[0][1])
        plt_pred_x_lst.append(i[1][0])
        plt_pred_y_lst.append(i[1][1])
    plt.plot(plt_org_x_lst,plt_org_y_lst,label="Pravá trajektorie")
    plt.plot(plt_pred_x_lst,plt_pred_y_lst,label ="Detekovaná trajektorie")
    plt.legend()
    plt.title("Trajektorie pohybu špičky jehly")
    plt.ylabel("Y[Pixelů]")
    plt.xlabel("X[Pixelů]")
    plt.savefig(f"plot_dir/512_512_b8_trajectory_resnet34.pdf")
    plt.show()


    plt.show()
if __name__ == '__main__':
    model = load_model("data/weightsfile_(1024, 1024)_resnet34_batch_10.h5")

    obj = Cvat_manipulator(
        "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset\\images",size=img_size)

    iterator = iter(obj)
    elem = next(iterator)

    for i in range(400):
        elem = next(iterator)
    lenght = elem.get_lenght()

    lst_dst = []
    trace_lst = []
    for i in tqdm(range(lenght)):
        try:

            elem = next(iterator)
            point = elem.get_point()
            point = point["needle_holder"]
            img, np_img = elem.get_img()

            fin_point = (0,0)

            prediction = predict(np_img, model,plot=False)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            oper_lst = []
            old_dst = 1000

            # visualiation part
            if point != None:
                cv2.circle(np_img, (int(point[0]), int(point[1])), radius=5, color=(0, 0, 255), thickness=-1)
            for i in prediction:
                cv2.circle(np_img, (int(i[1]), int(i[0])), radius=5, color=(0, 255, 0), thickness=-1)
                i = (i[1],i[0])
                dst = calc_euklid_distance(i, point)
                if dst<old_dst:
                        fin_point = i

                oper_lst.append(dst)
            if(fin_point == None):
                fin_point = trace_lst[-1]
            trace_lst.append((fin_point,point))


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

    plot_trajectory(trace_lst)

    np.save(f"data/plots/lst_(1024, 1024)_resnet34_batch_1",lst_dst)

    plt.plot(lst_dst, label=f"")
    plt.title(f"Euklidovská vzdálenost od nástroje")
    plt.ylabel("Chyba")
    plt.xlabel("Snímek")
    plt.legend()
    #plt.savefig(f"plot_dir/512_512__b8_resnet34.pdf")
    plt.show()
