# import sys
import os
# import tensorflow as tf
# import pathlib
import random
import numpy as np


# def preprocess_image(image):
#     # image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.decode_png(image, channels=1)
#     image = tf.image.resize(image, [192, 192])
#     image /= 255.0  # normalize to [0,1] range
#
#     return image
#
#
# def load_and_preprocess_image(path):
#     image = tf.read_file(path)
#     return preprocess_image(image)
#
#
# def get_filepaths(dirpath):
#     """
#     Return paths of all files in a directory.
#     :param dirpath:
#     :return:
#     """
#     data_root = pathlib.Path(dirpath)
#     all_image_paths = list(data_root.glob('*'))
#     all_image_paths = [str(path) for path in all_image_paths]
#     random.shuffle(all_image_paths)
#
#     return all_image_paths
#
#
# def load_dataset(dirpath):
#
#     all_image_paths = get_filepaths(dirpath)
#     path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
#     image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     return image_ds

def train_test_split(path, test_frac = 0.2):
    """
    Split images in directory into train and test subdirectories.
    :param path:
    :return:
    """

    # Get a list of all image files
    images = [f for f in os.listdir(path) if not f.startswith('.')]

    # Get indices of train and test indices
    test_size = int(test_frac * len(images))

    images_test = random.sample(images,test_size)
    images_train = list(set(images)-set(images_test))
    print(len(images_test))
    print(len(images_train))

    # Create train and test subfolders
    path_test = os.path.join(path,'test')
    path_train = os.path.join(path, 'train')
    if not os.path.exists(path_test):
        print("Creating test directory")
        os.makedirs(path_test)
        # Move images to subfolders
        for im in images_test:
            os.rename(os.path.join(path, im), os.path.join(path_test, im))
    if not os.path.exists(path_train):
        print("Creating train directory")
        os.makedirs(path_train)
        for im in images_train:
            os.rename(os.path.join(path, im), os.path.join(path_train, im))


def get_images_mean(image_dataset):
    """
    Get mean of image tensors.
    :param image_dataset:
    :return:
    """

    samples = []
    for i in range(len(image_dataset)):
        sample, _ = image_dataset[i]
        samples.append(sample.numpy())

    arr = np.array(samples)
    sample_mean = np.squeeze(np.mean(arr,axis=0))
    return sample_mean


