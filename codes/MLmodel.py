# Using convolutional neural network to make an ML model which recognizes alphabets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


train = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
from sklearn.utils import shuffle
train = shuffle(train)

x_train = train.drop('0', axis=1)
x_train /= 255.0
y_train = train['0']
# print(max(x_train.iloc[3]))
# plt.ylabel(y_train[15655])
# plt.imshow(x_train.iloc[15655].values.reshape(28,28),interpolation='nearest', cmap='Greys')
# plt.show()


my_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, "relu"),
    tf.keras.layers.Dense(26, "softmax")
])

x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
# y_train = tf.keras.utils.to_categorical(y_train)
my_model.compile(optimizer="adam",loss= "sparse_categorical_crossentropy", metrics="acc")

my_model.fit(x_train, y_train, epochs=1, shuffle=True, validation_split=0.2)

my_model.save('alphabet_classifier.h5')

