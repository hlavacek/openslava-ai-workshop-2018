#!/usr/bin/env python

import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np


def load():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    print("The MNIST database has a training set of %d examples." % len(train_data))
    print("The MNIST database has a test set of %d examples." % len(test_data))

    return (train_data, train_labels), (test_data, test_labels)


def show(x_train, y_train):
    # plot first six training images
    fig = plt.figure(figsize=(20, 20))
    for i in range(6):
        ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
        ax.imshow(x_train[i], cmap='gray')
        ax.set_title(str(y_train[i]))
    plt.show()


def show_detail(x_train):
    img = x_train[0]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    plt.show()


def rescale(train_data, test_data):
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255

    # train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    # test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    return train_data, test_data


def one_hot_encode(train_labels, test_labels):
    # print first ten (integer-valued) training labels
    print('Integer-valued labels:')
    print(train_labels[:10])

    # one-hot encode the labels
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    # print first ten (one-hot) training labels
    print('One-hot labels:')
    print(test_labels[:10])

    return train_labels, test_labels


def define_dense_model(train_data):
    # define the model
    K.clear_session()
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # summarize the model
    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def define_cnn_model():
    # define the model
    K.clear_session()
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                     input_shape=(28, 28, 1)))  # 28 x 28
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=28, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=56, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # summarize the model
    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',  # keras.optimizers.Adadelta()
                  metrics=['accuracy'])

    return model


def my_eval(model, data, labels):
    # evaluate test accuracy
    score = model.evaluate(data, labels, verbose=0)
    accuracy = 100 * score[1]

    # print test accuracy
    print('Test accuracy: %.4f%%' % accuracy)


def train(model, train_data, train_labels):

    # train the model
    print("train_data[1].shape {}".format(train_data[1].shape))

    checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5',
                                   verbose=1, save_best_only=True)

    hist = model.fit(train_data, train_labels, batch_size=128, epochs=10,
                     validation_split=0.2, callbacks=[checkpointer],
                     verbose=1, shuffle=True)


def predict(model, image):
    return int(model.predict(np.expand_dims(image, axis=0)).argmax())


def show_predictions(predictions):
    # plot first six training images
    fig = plt.figure(figsize=(20, 20))
    for i in range(len(predictions)):
        prediction, image = predictions[i]
        ax = fig.add_subplot(3, 9, i+1, xticks=[], yticks=[])
        ax.imshow(image, cmap='gray')
        ax.set_title(str(prediction))
    plt.show()


def dunno():
    (train_data, train_labels), (test_data_ori, test_labels_ori) = load()
    # show(x_train, y_train)
    # show_detail(x_train)
    train_data, test_data = rescale(train_data, test_data_ori)
    train_labels, test_labels = one_hot_encode(train_labels, test_labels_ori)

    # model = define_dense_model(train_data)
    model = define_cnn_model()
    my_eval(model, test_data, test_labels)
    # train(model, train_data, train_labels)
    model.load_weights("mnist.model.best.hdf5")
    my_eval(model, test_data, test_labels)
    predictions = [(predict(model, image), image_ori) for image, image_ori in zip(test_data[15:32], test_data_ori[15:32])]
    show_predictions(predictions)


if __name__ == "__main__":
    dunno()
