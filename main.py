import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
mnist = tf.keras.datasets.mnist
# y_binary = to_categorical(y_int)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train)

loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)
# print(accuracy)

model.save('digits.model')
