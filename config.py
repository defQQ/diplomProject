##Parametr train
HEIGHT = 48
WIDTH = 48
CHANNELS = 3
EPOCH = 50
BATCH_SIZE = 900


##Config view_augmentation.py
N_SAMPLES = 10
N_CLASS_EXAMPLE = 10

##Paths metainfo
r1_path_folder = 'classific/rtsd-r1.tar/rtsd-r1/'
train_csv = 'classific/rtsd-r1.tar/rtsd-r1/gt_train.csv'
icons_folder = 'classific/sign-icons'
path_numbers_to_classes = 'classific/rtsd-r1.tar/rtsd-r1/numbers_to_classes.csv'

##Defaul path to momels
lenet = 'lenet.hdf5'
alexnet = 'alexnet.hdf5'
vgg16 = 'vgg16.hdf5'
vgg19 = 'vgg19.hdf5'
xception = 'xception.hdf5'
