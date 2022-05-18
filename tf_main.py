import random
import segmentation_models as sm
import matplotlib.pyplot as plt
import PIL
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
from PIL import ImageOps
from tensorflow import keras
import numpy as np
import tensorflow as tf
from cvat_manipulator import Cvat_manipulator

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

input_dir = "data/dataset_1/segmentation_dataset/images"
target_dir = "masks/"
sm.set_framework('tf.keras')
img_size = (1024, 1024)
batch_size = 1
epochs = 2
i = 4


class DataLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, cvat_manipulator):
        self.batch_size = batch_size
        self.img_size = img_size
        self.cvat_manipulator = cvat_manipulator

    def __len__(self):
        return self.cvat_manipulator.get_lenght() // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        mask_lst = []
        img_lst = []
        for i in range(i, i + self.batch_size):
            mask_lst.append(self.cvat_manipulator.get_mask(i))
            img_lst.append(self.cvat_manipulator.get_img(i)[1])

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for counter, j in enumerate(img_lst):
            x[counter] = j

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for counter, j in enumerate(mask_lst):
            z = np.expand_dims(j, 2)
            y[counter] = z
        return x, y


BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

cvat = Cvat_manipulator("data/dataset_1/segmentation_dataset/annotations.xml",
                        "data/dataset_1/segmentation_dataset/images"
                        , img_size, shuffle=True)

cvat.add_dataset(
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_3\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_3\\images", shuffle=True)
cvat.add_dataset(
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_4\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_4\\images", shuffle=True)
cvat.add_dataset(
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_5\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_5\\images", shuffle=True)
cvat.add_dataset(
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_6\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_6\\images", shuffle=True)
cvat.add_dataset(
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_7\\annotations.xml",
    "C:\\Users\\David\\PycharmProjects\\bakalarka\\data\\dataset_1\\segmentation_dataset_7\\images", shuffle=True
)

cvat_train, cvat_val = cvat.split(0.005)

train_gen = DataLoader(
    batch_size, img_size, cvat_train
)

val_gen = DataLoader(
    batch_size, img_size, cvat_val
)

# define model
model = Unet(BACKBONE, input_shape=(img_size[0], img_size[1], 3), encoder_weights='imagenet', classes=2)
model.compile('Adam', loss="sparse_categorical_crossentropy", metrics=["accuracy", iou_score])
tf.keras.utils.plot_model(model, to_file='model.pdf', dpi=600,
                          expand_nested=True, show_shapes=True)
checkpoint_filepath = '/data/checkpoint/model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    verbose=True,
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq="epoch",
    save_best_only=True)


def display_mask():
    i = random.randint(0, val_gen.cvat_manipulator.get_lenght() - 1)
    val_preds = model.predict(val_gen)
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))

    _, np_img = cvat_val.get_img(i)
    plt.imshow(img, interpolation='nearest')
    plt.imshow(np_img, interpolation='nearest', alpha=0.6)
    plt.show()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        display_mask()
        print("epoch")


# fit model
history = model.fit(train_gen, validation_data=val_gen, callbacks=[model_checkpoint_callback, DisplayCallback()],
                    epochs=epochs)

model.save_weights(f'data/weightsfile{str(img_size)}_{BACKBONE}_batch_{batch_size}.h5')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
