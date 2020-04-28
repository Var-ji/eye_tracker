import numpy as np 
import cv2
from sklearn.utils import shuffle
import argparse
import os

import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers


probabilities = {'down':np.array([1, 0, 0, 0, 0], dtype=np.float32),
                 'left':np.array([0, 1, 0, 0, 0], dtype=np.float32),
                 'center':np.array([0, 0, 1, 0, 0], dtype=np.float32),
                 'right':np.array([0, 0, 0, 1, 0], dtype=np.float32),
                 'up':np.array([0, 0, 0, 0, 1], dtype=np.float32)}

def createModel():
    # base_model = MobileNetV2(input_shape=(224, 224, 3), # Original image shape = (200, 120, 3) (w, h, d)
    #                          weights=None,
    #                          include_top=False)
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # predictions = Dense(5, activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=predictions)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    return model

def loadTrainingData(data_path='./training_data'):
    train_classes = ['down', 'left', 'center', 'right', 'up']
    X = []
    Y = []
    for direction in train_classes:
        folder_path = os.path.join(data_path, direction)
        y = probabilities[direction]
        for root, _, files in os.walk(folder_path):
            for image in files:
                im_path = os.path.join(root, image)
                imfile = cv2.imread(im_path)
                assert imfile.shape == (120, 200, 3), "Input image shape is not of the required size"
                imfile = cv2.resize(imfile, (224, 224), interpolation=cv2.INTER_CUBIC)
                X.append(imfile)
                Y.append(y)
        print("Processed directory : %s" % (direction))

    X, Y = shuffle(X, Y, random_state=666)

    return X, Y

def trainModel(data_path='./training_data'):
    X, Y = loadTrainingData(data_path)
    model = createModel()
    opt = optimizers.RMSprop(lr=0.0001, rho=0.9, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    X = np.asarray(X)
    Y = np.asarray(Y)

    model.fit(X, Y, validation_split=0.10, epochs=25, batch_size=64, shuffle=True)
    model.save('./eye_tracker.h5')

    print("Done")


if __name__ == '__main__':
    trainModel()
