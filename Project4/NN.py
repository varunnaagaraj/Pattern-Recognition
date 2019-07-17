# Created by Varun at 17/04/19
import gzip
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from keras import optimizers
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
minst_training_data, minst_validation_data, minst_test_data = pickle.load(f)
f.close()


def neural_net_implementation_part1(dataset, minst_testing):
    """
    Neural Network Implementation
    :param dataset: Dataset
    :param minst_testing: MNIST Test data
    :param USPS_mat: USPS Feature Matrix
    :param USPS_target: USPS Test Matrix
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical, plot_model
    from keras import backend

    def rmse(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    image_vector_size = 28 * 28
    x_train = dataset[0].reshape(dataset[0].shape[0], image_vector_size)
    x_test = minst_testing[0].reshape(minst_testing[0].shape[0], image_vector_size)
    y_train = to_categorical(dataset[1], 10)
    y_test = to_categorical(minst_testing[1], 10)
    x_train = x_train[:1000, :]
    x_test = x_test[:1000, :]
    y_train = y_train[:1000]
    y_test = y_test[:1000]
    print(y_train.shape)
    print(x_train.shape)
    model = Sequential()
    model.add(Dense(30, input_shape=(784,), activation='sigmoid'))

    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy', rmse])
    plot_model(model, 'model.png')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    history = model.fit(x_train, y_train, batch_size=10, validation_split=0.2, callbacks=[monitor], verbose=0,
                        epochs=30)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('error.png')
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('criterion')
    plt.ylabel('training')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('criterion.png')

    print("MNIST")
    score, acc, rmserror = model.evaluate(x_test, y_test)
    print("Score (RMSE) : {}".format(score))
    print("accuracy is :{}".format(acc))


def neural_net_implementation_part2ab(dataset, minst_testing):
    """
    Neural Network Implementation
    :param dataset: Dataset
    :param minst_testing: MNIST Test data
    :param USPS_mat: USPS Feature Matrix
    :param USPS_target: USPS Test Matrix
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical, plot_model
    from keras import backend

    def rmse(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    image_vector_size = 28 * 28
    x_train = dataset[0].reshape(dataset[0].shape[0], image_vector_size)
    x_test = minst_testing[0].reshape(minst_testing[0].shape[0], image_vector_size)
    y_train = to_categorical(dataset[1], 10)
    y_test = to_categorical(minst_testing[1], 10)
    x_train = x_train[:1000, :]
    x_test = x_test[:1000, :]
    y_train = y_train[:1000]
    y_test = y_test[:1000]
    print(y_train.shape)
    print(x_train.shape)
    model = Sequential()
    model.add(Dense(30, input_shape=(784,), activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    # model.add(Dense(30, activation='sigmoid', activity_regularizer=l2(5)))
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', rmse])
    plot_model(model, 'model2a2.png')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    history = model.fit(x_train, y_train, batch_size=10, validation_split=0.2, callbacks=[monitor], verbose=0,
                        epochs=30)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('error2a2.png')
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy2a2.png')
    plt.clf()
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('criterion')
    plt.ylabel('training')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('criterion2a2.png')

    print("MNIST")
    score, acc, rmserror = model.evaluate(x_test, y_test)
    print("Score (RMSE) : {}".format(score))
    print("accuracy is :{}".format(acc))


def neural_net_implementation_part3ab(dataset, minst_testing):
    """
    Neural Network Implementation
    :param dataset: Dataset
    :param minst_testing: MNIST Test data
    :param USPS_mat: USPS Feature Matrix
    :param USPS_target: USPS Test Matrix
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical, plot_model
    from keras import backend

    def rmse(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    image_vector_size = 28 * 28
    x_train = dataset[0].reshape(dataset[0].shape[0], image_vector_size)
    x_test = minst_testing[0].reshape(minst_testing[0].shape[0], image_vector_size)
    y_train = to_categorical(dataset[1], 10)
    y_test = to_categorical(minst_testing[1], 10)
    x_train = x_train[:1000, :]
    x_test = x_test[:1000, :]
    y_train = y_train[:1000]
    y_test = y_test[:1000]
    print(y_train.shape)
    print(x_train.shape)
    model = Sequential()
    model.add(Dense(30, input_shape=(784,), activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    # model.add(Dense(30, activation='sigmoid', activity_regularizer=l2(5)))
    # model.add(Dense(30, activation='sigmoid', activity_regularizer=l2(5)))
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', rmse])
    plot_model(model, 'model3a.png')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    history = model.fit(x_train, y_train, batch_size=10, validation_split=0.2, callbacks=[monitor], verbose=0,
                        epochs=30)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('error3a.png')
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy3a.png')
    plt.clf()
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('criterion')
    plt.ylabel('training')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('criterion3a.png')

    print("MNIST")
    score, acc, rmserror = model.evaluate(x_test, y_test)
    print("Score (RMSE) : {}".format(score))
    print("accuracy is :{}".format(acc))


def augment_and_train():
    from keras.datasets import mnist
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras import backend as K
    K.set_image_dim_ordering('th')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    X_train = X_train[:1000, :]
    X_test = X_test[:1000, :]
    Y_train = Y_train[:1000]
    Y_test = Y_test[:1000]

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    BatchNormalization()
    model.add(Dense(512, activation='relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    gen = ImageDataGenerator(rotation_range=3, width_shift_range=0.1, height_shift_range=0.1)
    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=64)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

    model.fit_generator(train_generator, steps_per_epoch=1000 // 64, epochs=5,
                        validation_data=test_generator, validation_steps=1000 // 64)

    score = model.evaluate(X_test, Y_test)
    print('Test accuracy: ', score[1])
    predictions = model.predict_classes(X_test)

    predictions = list(predictions)
    actuals = list(Y_test)

    sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
    sub.to_csv('./output_cnn.csv', index=False)


neural_net_implementation_part1(minst_training_data, minst_test_data)
neural_net_implementation_part2ab(minst_training_data, minst_test_data)
neural_net_implementation_part3ab(minst_training_data, minst_test_data)
augment_and_train()
