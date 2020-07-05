from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input, AveragePooling2D
import utils
import config
from tensorflow import Tensor
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19


def lenet():
    """
       Build LeNet model
       :return: model - model leNet
    """
    input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(utils.n_classes_r1(), activation='softmax'))
    return model


def alexNet():
    """
        Build an AlexNet
        :return: model - model alexNet
    """

    input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(utils.n_classes_r1(), activation='softmax'))
    return model


def vgg16():
    """
    Build VGG16
    :return: model - model VGG16
    """
    input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)
    input_tensor = Input(shape=input_shape)

    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(utils.n_classes_r1(), activation='softmax'))

    model = Sequential()
    for l in base_model.layers:
        model.add(l)

    model.add(top_model)

    for layer in model.layers[:15]:
        layer.trainable = False

    return model


def xception():
    """
       Build Xception
       :return: model - model Xception
       """
    input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)

    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_tensor=Input(input_shape))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(utils.n_classes_r1(), activation='sigmoid'))
    return model


def vgg19():
    """
       Build VGG19
       :return: model - model VGG19
       """
    input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)

    base_model = VGG19(weights='imagenet',
                       include_top=False,
                       input_tensor=Input(input_shape))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(utils.n_classes_r1(), activation='sigmoid'))

    return model
