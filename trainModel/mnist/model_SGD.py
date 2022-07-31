import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from random import randint

# Data Loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Pre-processing

# Reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# Label Encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.25,
                                                    random_state=42)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

model = Sequential()

# Layer 1
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='Same',
                 input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Layer 2
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Fully Connected
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# Optimizer
optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

# Compile Model
model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

epochs = 10
batch_size = 250

# Model Training
train_model = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test))

# Model Save
model.save("../../mnist/savedM/model-8.h5")

# Score
score = model.evaluate(x_test, y_test, batch_size=batch_size)

print("Test Accuracy: ", score[1])