import pandas as pd
from keras.utils.np_utils import to_categorical
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import matplotlib.image as mpimg
from keras.models import load_model
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import config


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


def n_classes_r1():
    """
    Count classes of metainfo files
    :return: n_classes_r1(int) - count classes
    """
    try:
        classes_names_df_r1 = pd.read_csv(config.path_numbers_to_classes)
        n_classes_r1 = classes_names_df_r1['class_number'].count()
        return n_classes_r1
    except Exception as e:
        print("Ошибке чтения файла \n " + str(e))


def loading_saved_model(model_path):
    """
    Load model
    :param model_path: path to model
    :return: model
    """
    path_str=''
    path_str = path_str.join(model_path)
    return load_model(path_str)


def load_data(imgs_paths: list, num_labels: list = []):
    """

    :param imgs_paths:
    :param num_labels:
    :return:
    """
    data = []
    data_len = imgs_paths.shape[0]

    for i in range(data_len):
        image = cv2.imread(imgs_paths[i])
        sized_image = cv2.resize(image, (config.WIDTH, config.HEIGHT))
        sized_image = cv2.cvtColor(sized_image, cv2.COLOR_BGR2RGB)
        # sized_image = cv2.cvtColor(sized_image, cv2.COLOR_RGB2GRAY)

        data.append(img_to_array(sized_image))

    data = np.array(data, dtype="float") / 255.0

    if num_labels:
        labels = to_categorical(num_labels, num_classes=n_classes_r1())
        return data, labels
    else:
        return data


def visual_trainin(history, model):
    """
    Build graphics training model
    :param history: history train model
    :param model: load model
    :return:
    """
    fig0 = plt.figure(0)
    plt.style.use(['classic'])
    fig0.set_facecolor('w')
    plt.plot(history.history['acc'], label='Train accuracy')
    plt.plot(history.history['val_acc'], label='Val accuracy')
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.title('Accuracy during learning', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='best')
    fig0.savefig(model + '_train_acc.png')

    fig1 = plt.figure(1)
    plt.style.use(['classic'])
    fig1.set_facecolor('w')
    plt.plot(history.history['loss'], label='Train Loss-function')
    plt.plot(history.history['val_loss'], label='Val Loss-function')
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.title('Loss-function during learning', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss-function', fontsize=14)
    plt.legend(loc='best')
    fig1.savefig(model + '_train_loss.png')


def plot_add_icon(sign_class, df_classes, ax, xlims,
                  zoom: float = 0.45, delta_w: float = 500):
    """
    Add icon to plot info
    :param sign_class:
    :param df_classes: list classes
    :param ax: marking plot
    :param xlims: x range
    :param zoom: zoom icon
    :param delta_w: bias icon
    :return:
    """
    y_bar_nums = df_classes.index.astype(str)

    y_bar_labels = (('№' + y_bar_nums).to_numpy(dtype=str))
    x_bar_labels = df_classes[0].to_numpy()

    bar_graphs = ax.barh(y_bar_labels, x_bar_labels, 0.8,
                         alpha=0.8)

    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_xlim((xlims[0], xlims[1]))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=ymin - 1, top=ymax + 1)

    for i, bar in enumerate(bar_graphs):
        w, h = bar.get_width(), bar.get_height()

        x0, y0 = bar.xy

        path = config.icons_folder + '/' + str(sign_class[int(y_bar_nums[i])]) + '.gif'
        arr_img = mpimg.imread(path)
        imagebox = OffsetImage(arr_img, zoom=zoom)

        ab = AnnotationBbox(imagebox, (x0 + w + delta_w, y0 + h / 2), frameon=False)
        ax.add_artist(ab)

        label = round(x_bar_labels[i], 5)
        ax.text(x=x0 + w + 0.02, y=y0 + h / 4, s=label, size=10)
