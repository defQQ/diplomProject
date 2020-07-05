import argparse, utils, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import pandas as pd


def parse_args():
    """
        Parse arguments of command line
        :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',
                        help='path to model',
                        type=str, default='')
    args = parser.parse_args()

    return args


def check_path(args):
    """
    Check input user arguments
    :param args: user input args
    :return:
    """
    try:
        if not os.path.isfile(args.path):
            raise Exception('inccorrect path to model ')

    except Exception as e:
        print(e)


def confusion_matrix(model, train_X, train_Y, path_name):
    """
    Build confusion matrix
    :param model: model
    :param train_X: data for predict
    :param train_Y: label for predict
    :param path_name: path to name model
    :return:
    """
    y_pred = model.predict_classes(train_X)
    test_y_classes = np.argmax(train_Y, axis=1)

    con_mat1 = metrics.confusion_matrix(y_true=test_y_classes, y_pred=y_pred)
    con_mat_norm1 = np.around(con_mat1.astype('float') / con_mat1.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df1 = pd.DataFrame(con_mat_norm1, index=range(utils.n_classes_r1()),
                               columns=range(utils.n_classes_r1()))

    fig = plt.figure(figsize=(30, 30))
    ax = sb.heatmap(con_mat_df1, square=True,
                    annot=True, linecolor='silver',
                    cmap=plt.cm.Blues,
                    linewidths=0.75,
                    cbar=False)

    ax.tick_params(labelright=True, left=True, bottom=True, labeltop=True,
                   rotation=0, labelsize=16)

    plt.ylabel('True class number', fontsize=20)
    plt.xlabel('Obtained class number', fontsize=20)
    fig.savefig(path_name + '_confusion_matrix.png')


def evaluate_model(model, val_x, val_y):
    """
    Evaluate model
    :param model: model
    :param val_x: data for predict
    :param val_y: labels for predict
    :return:
    """
    val_eval = model.evaluate(val_x, val_y)
    print(f'loss: {round(val_eval[0], 5)}, acc: {round(val_eval[1], 5)}')


if __name__ == '__main__':
    args = parse_args()
    check_path(args)
    model = utils.loading_saved_model(args.path)
    num_labels, img_paths = utils.read_metainfo()
    imgs, labels = utils.load_data(img_paths, num_labels=num_labels)

    train_x, val_x, train_y, val_y = train_test_split(
        imgs, labels,
        test_size=0.4,
        random_state=20,
        stratify=labels)
    confusion_matrix(model, val_x, val_y, args.path)
    evaluate_model(model, val_x, val_y)
    print("Confusion matrix save to confusion_matrix.png ")
    print("Script completed successfully")
