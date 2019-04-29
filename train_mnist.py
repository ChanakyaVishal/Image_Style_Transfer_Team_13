from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import sys
import _pickle as cPickle
import os
import gzip
import tensorflow
import matplotlib.pyplot as plt


f = gzip.open('mnist.pkl.gz', 'rb')
data = cPickle.load(f, encoding='bytes')

(x_train, y_train), (x_test, y_test) = data

x_train = np.reshape(x_train.astype('float32'), (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test.astype('float32'), (len(x_test), 28, 28, 1)) 


def encoder():
    input_img = Input(shape=(28, 28, 1)) 

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    model = Model(input_img, encoded, name="encoder")
    return model





