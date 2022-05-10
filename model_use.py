
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
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from kalman_filter import KalmanFilter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
img_size =(800,800)

def load_model(path='data/weightsfile.h5'):
    sm.set_framework('tf.keras')
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    model = Unet(BACKBONE, input_shape=(img_size[0], img_size[1], 3), encoder_weights='imagenet', classes=2)
    model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=[iou_score])
    model.load_weights('data/weightsfile.h5')
    return model

def predict(frame,model,plot = False):
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

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
    VideoCap = cv2.VideoCapture('data\\video_test_3.mov')

    HiSpeed = 100
    ControlSpeedVar = 30  # Lowest: 1 - Highest:100
    debugMode = 1

    for i in range(500):
        VideoCap.read()
    while (True):
        # Read frame
        ret, frame = VideoCap.read()
        frame = cv2.resize(frame,img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect object
        centers = predict(frame,model,plot=True)

        # If centroids are detected then track them
        if (len(centers) > 0):
            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)
            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), (255, 0, 0), 2)
            # Update
            (x1, y1) = KF.update(centers[0])
            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 0, 255), 2)
            cv2.putText(frame, "Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (centers[0][0] + 15, centers[0][1] - 15), 0, 0.5, (0, 191, 255), 2)
        cv2.imshow('image', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break
        cv2.waitKey(HiSpeed - ControlSpeedVar + 1)