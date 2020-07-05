import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse, config, utils, sys


def parse_args():
    """
        Parse arguments of command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--pathDataset',
                        help='path to folder with images',
                        type=str, default='classific/rtsd-r1.tar/rtsd-r1/train/')
    parser.add_argument('--trainMeta',
                        help='path to metainfo csv',
                        type=str, default='classific/rtsd-r1.tar/rtsd-r1/gt_train.csv')
    parser.add_argument('--icon',
                        help='path to folder icon',
                        type=str, default='classific/sign-icons')
    parser.add_argument('--numberClasses',
                        help='path to metainfo csv',
                        type=str, default='classific/rtsd-r1.tar/rtsd-r1/numbers_to_classes.csv')
    parser.add_argument('--nSamples',
                        help='number show sapmles',
                        type=int, default='10')
    args = parser.parse_args()

    return args


def read_metainfo(args):
    """
        Read and preprocessing metainfo
        :return: class_counts_image(list) - list with labels class
                 sign_class(ndarray) - array with full image paths
                 n_classes - count classes
    """
    try:
        train_csv = pd.read_csv(args.trainMeta)
        train_csv['filename'] = args.pathDataset + train_csv['filename']
        class_counts_image = train_csv['class_number'].value_counts()
        classes_names_df = pd.read_csv(args.numberClasses)
        n_classes = classes_names_df['class_number'].count()
        sign_class = classes_names_df['sign_class'].tolist()
        return class_counts_image, sign_class, n_classes, train_csv
    except Exception as e:
        print("Ошибке чтения файла \n " + str(e))


def visual_classes_dataset(args, class_counts_image, sign_class):
    """
    Build plot with information classes dataset
    :param args: parametrs  entered by the user
    :param class_counts_image: count images
    :param sign_class: list classes
    :return:
    """
    y_bar_nums = class_counts_image.index.astype(str)

    y_bar_labels = (('№' + y_bar_nums).to_numpy(dtype=str))
    x_bar_labels = class_counts_image.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 27))
    plt.style.use(['classic'])
    fig.set_facecolor('w')
    bar_graphs = ax.barh(y_bar_labels, x_bar_labels, 0.8,
                         edgecolor='lightseagreen',
                         color='lightseagreen', alpha=0.8)

    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((xmin, xmax * 1.1))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=ymin - 1, top=ymax + 1)

    for i, bar in enumerate(bar_graphs):
        w, h = bar.get_width(), bar.get_height()
        x0, y0 = bar.xy

        path = args.icon + '/' + str(sign_class[int(y_bar_nums[i])]) + '.gif'
        arr_img = mpimg.imread(path)
        imagebox = OffsetImage(arr_img, zoom=0.45)

        ab = AnnotationBbox(imagebox, (x0 + w + 250, y0 + h / 2), frameon=False)
        ax.add_artist(ab)

        # display the number of images 20 points to the right of the bar's end
        label = x_bar_labels[i]
        plt.text(x=x0 + w + 20, y=y0 + h / 2 - 0.1, s=label, size=12)

    plt.title('Number of classes in RTSD-r1', color='k', fontsize=20)
    plt.ylabel('Traffic sign class number', color='k', fontsize=16)
    plt.xlabel('Number of images', color='k', fontsize=16)
    fig.savefig('visual_classes.png')
    print("Visual_classes_dataset save picture visual_classes.png")


def view_show_variety(args, class_counts_image, sign_class, train_csv):
    """
       Build plot with information classes dataset
       :param args: parametrs  entered by the user
       :param class_counts_image: count images
       :param sign_class: list classes
       :return:
    """
    fig = plt.figure(figsize=((args.nSamples + 2) * 0.5, (n_classes + 2) * 0.5))
    fig.subplots_adjust(left=0.1, right=1, bottom=0, top=0.97, hspace=0.005, wspace=0.005)
    plt.style.use(['classic'])
    fig.set_facecolor('w')

    for row_i in range(n_classes):

        temp_filter = train_csv["class_number"] == row_i
        temp = train_csv.where(temp_filter)['filename'].dropna().sample(n=args.nSamples, random_state=1)
        temp_imgs = utils.load_data(temp.to_numpy(), class_counts_image)

        for col_i in range(args.nSamples):
            ax = fig.add_subplot(n_classes, args.nSamples,
                                 row_i * 10 + col_i + 1, xticks=[], yticks=[])
            ax.imshow(temp_imgs[col_i])
            if col_i == 0:
                row_text = '№' + str(row_i)

                ax.set_ylabel(row_text, rotation=0, labelpad=20)

    fig.suptitle('Examples of images per each class', color='k', fontsize=18)
    plt.tight_layout()
    fig.savefig('view_show_variety.png')
    print("view_show_variety save picture view_show_variety.png")


if __name__ == '__main__':
    args = parse_args()
    class_counts_image, sign_class, n_classes, train_csv = read_metainfo(args)
    visual_classes_dataset(args, class_counts_image, sign_class)
    view_show_variety(args, class_counts_image, sign_class, train_csv)
    print("view_show_variety save picture view_show_variety.png")
    print("Script completed successfully")
