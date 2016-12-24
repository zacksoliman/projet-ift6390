from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import numpy as np




#softmax, pas de couche cacher
def NN1(shapeinput, optim):
    model = Sequential()
    model.add(Dense(3, input_dim=shapeinput))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model


def NN2(shapeinput,optim,hiddenlayer,nbhid):
    model = Sequential()
    model.add(Dense(nbhid, input_dim=shapeinput))
    model.add(Activation(hiddenlayer))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model


def NN3(shapeinput,optim,hiddenlayer):
    model = Sequential()
    model.add(Dense(10, input_dim=shapeinput, init='uniform'))
    model.add(Activation(hiddenlayer))
    model.add(Dropout(0.5))
    #model.add(Dense(64, init='uniform'))
    #model.add(Activation(hiddenlayer))
    #model.add(Dropout(0.5))
    model.add(Dense(3, init='uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model






def NN100():

    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    #model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

    return model




def NN200(shapeinput):

    data_dim = shapeinput
    timesteps = 8
    nb_classes = 3
    batch_size = 32

    # expected input batch shape: (batch_size, timesteps, data_dim)
    # note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True, stateful=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # generate dummy training data
    x_train = np.random.random((batch_size * 10, timesteps, data_dim))
    y_train = np.random.random((batch_size * 10, nb_classes))

    # generate dummy validation data
    x_val = np.random.random((batch_size * 3, timesteps, data_dim))
    y_val = np.random.random((batch_size * 3, nb_classes))
    return model
