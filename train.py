import argparse
import sys
sys.path.append('../')
from src.model import get_model, freeze_all_but_mid_and_top_Dense, freeze_all_but_top_Dense
from src.log_cf_matrix import LogConfusionMatrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint
from src.dynamic_sampling import DynamicSamplingImageDataGenerator, BatchSizeAdapter
import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0},intra_op_parallelism_threads=8)))
os.environ['CUDA_VISIBLE_DEVICES']='/GPU:0'
def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Sampling training on Inception-v3-based model')

    data_group = parser.add_argument_group('data')
    data_group.add_argument('--num_class', type=int,default=10,
                            help='the number of classes in the dataset')
    data_group.add_argument('--train_path', type=str,default='F:\\原版cifar10\\cifar10\\train',
                            help='path to the directory of training images')
    data_group.add_argument('--valid_path', type=str,default='F:\\原版cifar10\\cifar10\\validate',
                            help='path to the directory of validation images')
    data_group.add_argument('--img_size', nargs=2, type=int, metavar=('img_height', 'img_width'), default=(128, 128),
                            help='the target size of input images')
    data_group.add_argument('--valid_batch', type=int, default=32,
                            help='batch size during validation')
    data_group.add_argument('--num_sample_per_class_per_batch', type=int, default=6,#定义每一类数量
                            help='defines batch size per class, batch_size = num_sample_per_class_per_batch * num_class')

    augment_group = parser.add_argument_group('augment')
    augment_group.add_argument('--shear_range', type=float, default=0.3)
    augment_group.add_argument('--horizontal_flip', type=bool, default=True)
    augment_group.add_argument('--rotation_range', type=float, default=10.)
    augment_group.add_argument('--width_shift_range', type=float, default=0.3)
    augment_group.add_argument('--height_shift_range', type=float, default=0.3)

    model_group = parser.add_argument_group('model_training')
    model_group.add_argument('--weight_path', type=str, default=None,
                             help='path to the model weight file')
    model_group.add_argument('--epoch', type=int, default=100,
                             help='the number of training epoch')
    model_group.add_argument('--log_path', type=str, default='F:\\原版cifar10\\cifar10\\log_path.txt',
                             help='path to the log file of training process')
    model_group.add_argument('--cflog_path', type=str, default='F:\\原版cifar10\\cifar10\\cflog_path.txt',
                             help='path to the log file of confusion matrix')
    model_group.add_argument('--cflog_interval', type=int, default=10,
                             help='frequency to log confusion matrix for the whole validation dataset')
    model_group.add_argument('--checkpoint_path', type=str, default=None,
                             help='path to store checkpoint model files')


    warmup_group = parser.add_argument_group('warmup')
    warmup_group.add_argument('--warmup', action='store_true', default=True,
                              help='set to train the last two layers as warmup process')
    warmup_group.add_argument('--warmup_epoch', type=int, default=5,
                              help='the number of warmup training, valid only when warmup option is used')
    args = parser.parse_args()
    return args


def main(args):
    print('Preparing Model...')
    model = get_model(args.num_class)
    if args.weight_path is not None:
        # Continue the previous training
        print('Loading saved model: \'{}\'.'.format(args.model_weights))
        model.load_weights(args.model_weights)

    print('Preparing Data...')
    datagen_train = DynamicSamplingImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.3,
        height_shift_range=0.3)
    datagen_train = datagen_train.flow_from_directory(
        args.train_path,
        target_size=(args.img_size[0], args.img_size[1]),
        class_size_per_batch=args.num_sample_per_class_per_batch)
    datagen_valid = ImageDataGenerator(rescale=1./255).flow_from_directory(
        args.valid_path,
        target_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.valid_batch)

    if args.warmup:
        print("Warm up...")
        model = freeze_all_but_top_Dense(model)
        model.fit_generator(
            datagen_train,
            steps_per_epoch=20,
            validation_data=datagen_valid,
            validation_steps=10,
            shuffle=True,
            epochs=args.warmup_epoch)

    print('Model Training...')
    callbacks = []
    if args.log_path is not None:
        callbacks.append(CSVLogger(args.log_path))
    if args.checkpoint_path is not None:
        callbacks.append(ModelCheckpoint(filepath=args.checkpoint_path, verbose=1, save_best_only=False))
    if args.cflog_path is not None:
        callbacks.append(LogConfusionMatrix(args.cflog_path, args.valid_path,
                                            args.cflog_interval, (args.img_size[0], args.img_size[1])))
    callbacks.append(BatchSizeAdapter((datagen_train, datagen_valid), len(datagen_valid)))

    model = freeze_all_but_mid_and_top_Dense(model)
    model.fit_generator(
        datagen_train,
        steps_per_epoch=20,
        validation_data=datagen_valid,
        validation_steps=10,
        shuffle=True,
        epochs=args.epoch,
        callbacks=callbacks)
    ##model.save('my_model.h5')

if __name__ == '__main__':
    args = parse_args()
    main(args)
