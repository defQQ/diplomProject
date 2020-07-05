from keras.preprocessing.image import ImageDataGenerator
import config, utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def image_augmentation():
    """
    Sets up augmentation
    :return: datagen(object) - object class ImageDataGenerator
    """
    datagen = ImageDataGenerator(
        rotation_range=8,
        zoom_range=[0.8, 1.2],
        width_shift_range=0.12,
        height_shift_range=0.12,
        brightness_range=[0.3, 1.7],
        fill_mode="nearest")
    return datagen


def read_metainfo():
    """
    Read and preprocessing metainfo
    :return: r1_class_counts - class counts

    """
    try:
        train_csv = pd.read_csv(config.train_csv)
        train_csv['filename'] = config.r1_path_folder + 'train/' + train_csv['filename']
        r1_class_counts = train_csv['class_number'].value_counts()
        return r1_class_counts, train_csv
    except Exception as e:
        print("Ошибке чтения файла \n " + str(e))

def view_augmentation():
    """
    Data augmentation example
    :return:
    """
    r1_class_counts, labels_df = read_metainfo()
    temp_classes = r1_class_counts.tail(config.N_CLASS_EXAMPLE).index.to_numpy()
    datagen = image_augmentation()

    fig, axes = plt.subplots(nrows=config.N_CLASS_EXAMPLE, ncols=config.N_SAMPLES + 1,
                             figsize=((config.N_SAMPLES + 2) * 0.5, (config.N_CLASS_EXAMPLE + 1 + 2) * 0.5))

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, hspace=0.01, wspace=0)
    plt.style.use(['classic'])
    fig.set_facecolor('w')

    for row_i in range(config.N_CLASS_EXAMPLE):
        temp_filter = labels_df["class_number"] == temp_classes[row_i]
        temp = labels_df.where(temp_filter)['filename'].dropna().sample(n=1, random_state=1)
        temp_imgs = utils.load_data(temp.to_numpy())

        aug_iterator = datagen.flow(temp_imgs, batch_size=config.N_SAMPLES)

        for col_i in range(0, config.N_SAMPLES + 1):
            ax = axes[row_i, col_i]
            ax.set_xticks([])
            ax.set_yticks([])

            if col_i == 0:
                ax.imshow(temp_imgs[0])
                row_text = '№' + str(temp_classes[row_i])
                ax.set_ylabel(row_text, rotation=0, labelpad=20)

            else:
                aug_img = aug_iterator.next()
                ax.imshow(np.uint8(aug_img[0]))

    plt.suptitle('Data augmentation', fontsize=16)
    fig.savefig('data_augumentation.png')
    print("View_augmentation save info to data_augumentation.png")


if __name__ == '__main__':
    view_augmentation()
    print("Script completed successfully")
