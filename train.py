import modelNN
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils, config
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import argparse, os
from datetime import datetime
import keras
from keras.optimizers import SGD


def parse_args():
    """
    Parse arguments of command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='name model,supported: lenet, alexnet, vgg16, vgg19, xception',
                        type=str, default='lenet')
    parser.add_argument('--augmentation',
                        help='True or False',
                        type=bool, default=False)
    args = parser.parse_args()

    return args


def check_path(args):
    """
    Check input user arguments
    :param args: user input args
    :return:
    """
    try:
        if not os.path.isfile(args.model):
            raise Exception('inccorrect path to model ')
    except Exception as e:
        print(e)


def compile_model(model):
    """
    Model compilation
    :param model: model neural network
    :return: model: model
    """
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


def read_metainfo():
    """
    Read and preprocessing metainfo
    :return: num_labels(list) - list with labels class
             img_paths(ndarray) - array with full image paths
    """
    try:
        train_csv = pd.read_csv(config.train_csv)
        train_csv['filename'] = config.r1_path_folder + 'train/' + train_csv['filename']
        num_labels = train_csv['class_number'].to_list()
        img_paths = train_csv['filename'].to_numpy()
        return num_labels, img_paths
    except Exception as e:
        print("Ошибке чтения файла \n " + str(e))


def training(model, augmentation):
    """
    Training selected model
    :param model: name supported model for trainin
    :param augmentation: training within/with augmentation, boolean value
    :return:
    """
    augmentation = False
    if model == 'lenet':
        if os.path.isfile('lenet.hdf5'):
            model = utils.loading_saved_model('lenet.hdf5')
        else:
            model = compile_model(modelNN.lenet())
            name_model = 'lenet'
    if model == 'alexnet':
        if os.path.isfile('alexnet.hdf5'):
            model = utils.loading_saved_model('alexnet.hdf5')
        else:
            model = compile_model(modelNN.lenet())
            name_model = 'alexnet'
    if model == 'vgg16':
        if os.path.isfile('vgg16.hdf5'):
            model = utils.loading_saved_model('vgg16.hdf5')
        else:
            model = compile_model(modelNN.lenet())
            name_model = 'vgg16'
    if model == 'xception':
        if os.path.isfile('xception.hdf5'):
            model = utils.loading_saved_model('xception.hdf5')
        else:
            model = compile_model(modelNN.lenet())
            name_model = 'xception'
    if model == 'vgg19':
        if os.path.isfile('vgg19.hdf5'):
            model = utils.loading_saved_model('vgg19.hdf5')
        else:
            model = compile_model(modelNN.lenet())
            name_model = 'vgg19'
    num_labels, img_paths = utils.read_metainfo()
    imgs, labels = utils.load_data(img_paths, num_labels=num_labels)

    train_x, val_x, train_y, val_y = train_test_split(
        imgs, labels,
        train_size=0.8,
        test_size=0.2,
        random_state=20,
        stratify=labels)
    save_model_path = args.model + '.hdf5'
    checkpoint_callback = ModelCheckpoint(filepath=save_model_path,
                                          monitor='val_loss',
                                          save_best_only=True)

    # logdir = "logs/fit/ " + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/fit/ " + name_model
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    if augmentation:
        datagen = ImageDataGenerator(
            rotation_range=8,
            zoom_range=[0.7, 1.4],
            width_shift_range=0.12,
            height_shift_range=0.12,
            brightness_range=[0.3, 1.7],
            fill_mode="nearest")
        history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=config.BATCH_SIZE),
                                      # batch_size=config.BATCH_SIZE,
                                      steps_per_epoch=train_x.shape[0],
                                      epochs=config.EPOCH,
                                      validation_data=(val_x, val_y),
                                      callbacks=[checkpoint_callback, tensorboard_callback])
    else:
        print("not aug")
        history = model.fit(train_x, train_y, batch_size=config.BATCH_SIZE, epochs=config.EPOCH,
                            validation_data=(val_x, val_y),
                            callbacks=[checkpoint_callback, tensorboard_callback])

    utils.visual_trainin(history, args.model)


if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    training(args.model, bool(args.augmentation))
    print("Model " + args.model + " successfully trained ")
