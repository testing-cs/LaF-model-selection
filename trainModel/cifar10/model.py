# Libraries
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import matplotlib.pyplot as plt

# CIFAR_10 is a set of 60K images, 32x32 pixels on 3 channels
INPUT_SHAPE = (32, 32, 3)  # for tensorflow as backend
#INPUT_SHAPE = (3, 32, 32) # for theano as backend
BATCH_SIZE = 128
EPOCHS = 20
CLASSES = 10

#load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# print('X_train shape: ', X_train.shape)
# print('Y_train shape: ', Y_train.shape)
# print('X_test shape: ', X_test.shape)
# print('Y_test shape: ', Y_test.shape)

# Convert to categorical
Y_train = np_utils.to_categorical(Y_train, CLASSES)
Y_test = np_utils.to_categorical(Y_test, CLASSES)

# Normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Define model
def model(input_shape, num_classes):
    X_input = Input(input_shape)
    
    X = Conv2D(32, (3,3), padding='same')(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    X = Dropout(0.25)(X)
    
    X = Conv2D(64, (3,3), padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    X = Dropout(0.25)(X)
    
    X = Flatten()(X)
    X = Dense(512)(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(num_classes)(X)
    X = Activation('softmax')(X)
    
    model = Model(inputs=X_input, outputs=X)
    
    return model

model = model(INPUT_SHAPE, CLASSES)

# Train model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

# Evaluate model
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
print('Test accuracy: ', score[1])

# Save model
model.save("../../cifar10/savedM/model-20.h5")

