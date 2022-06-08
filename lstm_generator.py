import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 8
time_size = 8

img_size = (512,512)

def replace_submatrix(mat, ind1, ind2, surroundings=10, val=-1):
    ind1 = round(ind1)
    ind2 = round(ind2)

    for i in range(surroundings):
        try:
            if (val == -1):
                rand_val = random.random()
                mat[ind1 - surroundings - i:ind1 + surroundings - i,
                ind2 - surroundings:ind2 + surroundings - i] = random.random()
            else:
                rand_val = random.uniform(0.85, 1)
                mat[ind1 - surroundings - i:ind1 + surroundings - i,
                ind2 - surroundings:ind2 + surroundings - i] = rand_val
        except:
            _ = 1
    return mat

class In_time_gen():
    def __init__(self,init_pos = (256, 256),size =(512,512)):
        self.pos = init_pos
        self.size = size

    def gen_matrix(self):

        random.seed()
        matrix = np.zeros(self.size)
        for i in range(3):
            x = random.randint(0, self.size[1])
            y = random.randint(0, self.size[1])
            matrix = replace_submatrix(matrix, x, y)

        replace_submatrix(matrix, self.pos[0], self.pos[1], val=1)
        self.matrix = matrix
        return matrix

    def gen_clear_matrix(self):
        matrix = np.zeros(self.size)
        replace_submatrix(matrix, self.pos[0], self.pos[1], val=1)
        return matrix

    def update_pos(self):
        self.pos = [self.pos[0] + random.randint(-20, 20), self.pos[1] + random.randint(-20, 20)]

    def get_pos(self):
        return (self.pos[0]/self.size[0],self.pos[1]/self.size[1])


def gen_test_data():
    matrix_lst = []
    pos = (400,600)
    for i in range(8):
        matrix = np.zeros((1024, 1024))
        replace_submatrix(matrix, pos[0], pos[1], val=1)
        pos = (pos[0]+10,pos[1]+10)
        matrix_lst.append(matrix)
    return matrix_lst

def gen_model():
    inp = layers.Input(shape=(8,1,512,512))
    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=512,
        kernel_size=(10, 10),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=512,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)


    model = keras.models.Model(inp, x)
    model.summary()
    return  model



model = gen_model()

model.compile(
    loss=tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(),
)

dataX = []
dataY = []

x = np.empty((100,8,1,512,512))
y = np.empty((100,8,1,512,512))
obj = In_time_gen()
for j in range(100):
    oper_datay=[]
    for i in range(8):
        x[j][i] = obj.gen_matrix()
        obj.gen_matrix()
        y[j][i] = obj.gen_clear_matrix()



model.fit(x,y,epochs=4,batch_size = 4)

data = gen_test_data()

x = np.empty((1,8,1,1024,1024))


for idx,i in enumerate (data):
    x[0][idx] = i

y = model.predict(x)
print(y[0][0]*1024,y[0][1]*1024)
print("done")