from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
import warnings
import numpy as np
from keras.layers.normalization import BatchNormalization


class EarlyStoppingByAccuracy(keras.callbacks.Callback):
    def __init__(self, monitor='val_acc', value=0.9960, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


# load data
# since it needs vpn to download mnist from the official url
# we just download mnist,npz from a website and then put it at the same folder as the .py
def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# batchsize and epochs
batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#Normalize the
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Choose the random seeds
seed = 6
# split 15% of the traning images out for validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.15, random_state=seed)

# Define the sequential model
model = Sequential()

model.add(Conv2D(filters = 8, kernel_size=(5,5),padding = 'same' ,input_shape = input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Dropout(0.5))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(5))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))
# set optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# reduce the learning_rate by a half
# if the validation accuracy does not change along 3 epochs, min learning rate=0.000005
learning_rate = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.000005)

# early stop training if the val_acc has been 99.6% to avoid outfiting
# early_stop = EarlyStoppingByAccuracy(monitor='val_acc', value=0.9960, verbose=1)

datagen = ImageDataGenerator(
        rotation_range=15,  # rotate images for 15 degree
        zoom_range = 0.2,  # zoom image X1.2
        width_shift_range=0.2,  # shift images horizontally
        height_shift_range=0.2)  # shift images vatically
datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate])
score = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', score[1])
model.save("../../mnist/savedM/model-9.h5")
