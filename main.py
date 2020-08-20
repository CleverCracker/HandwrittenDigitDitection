import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist
# y_binary = to_categorical(y_int)

type(mnist)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# flatten 28*28 images to a 784 vector for each image


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=200, verbose=2)

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)
# print(accuracy)

model.save('digits.model')

for x in range(1, 10):
    imgPath = f'images/{x}.png'
    img = cv.imread(imgPath)[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The Result = {np.argmax(prediction)}')

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
