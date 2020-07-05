import os
import utils, config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def parse_args():
    """
    Parse arguments of command line
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='path to model',
                        type=str)
    parser.add_argument('--img',
                        help='path to image or path to folder with images',
                        type=str)
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
        if not os.path.isfile(args.img):
            raise Exception('inccorrect path to image ')

    except Exception as e:
        print(e)


def get_info_for_predict(path_image):
    """
    Read metainfo and processing data
    :param path_image: path to image or images
    :return:
     n_images_predict(int) - number of image
     full_paths_images_predict(ndarray) - Array with data full paths to images
     sign_class(list) - list of predictable classes
    """
    numbers_to_classes = pd.read_csv(config.path_numbers_to_classes)
    sign_class = numbers_to_classes['sign_class'].tolist()
    names_images_predict = np.asarray(path_image, dtype=np.str)

    if names_images_predict.size == 1:
        n_images_predict = 1
        full_paths_images_predict = names_images_predict.reshape(1, )
    else:
        n_images_predict = names_images_predict.shape[0]
        full_paths_images_predict = names_images_predict
    return n_images_predict, full_paths_images_predict, sign_class


def predict(path_model, path_img):
    """
    Predict classes for images, show plot with information
    :param path_model: path to model
    :param path_img: path to image or images
    :return:
    """
    n_images_predict, full_paths_images_predict, sign_class = get_info_for_predict(path_img)
    model = utils.loading_saved_model(path_model)
    data_predict_image = utils.load_data(full_paths_images_predict)

    predict_imgs_outputs = model.predict(data_predict_image)

    fig, axes = plt.subplots(nrows=n_images_predict, ncols=2,
                             figsize=(8, n_images_predict * 2))

    fig.subplots_adjust(left=0, right=1, bottom=0.2, top=0.95, hspace=0.1, wspace=0.01)
    plt.style.use(['classic'])
    fig.set_facecolor('w')

    for i_row in range(n_images_predict):

        df_class = pd.DataFrame(predict_imgs_outputs[i_row],
                                index=range(utils.n_classes_r1()))
        df_class = df_class.sort_values(0, ascending=False).head(3)
        for i_col in range(2):
            ax = axes[i_row, i_col]
            if i_col == 0:
                ax.imshow(data_predict_image[i_row])
                # ax.set_ylabel(names_images_predict[i_row])
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                utils.plot_add_icon(sign_class, df_class, ax, xlims=[0, 2],
                                    delta_w=0.28, zoom=0.5, )
                if i_row == 0:
                    ax.tick_params(labeltop=True, labelbottom=False, labelsize=10)
                elif i_row == n_images_predict - 1:
                    ax.tick_params(bottom=True, labeltop=False, labelsize=10)
                else:
                    ax.set_xticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    predict(args.model, args.img)
