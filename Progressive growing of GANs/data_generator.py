'''
A data generator for images

Inspired from the work of Pierluigi Ferrari in : https://github.com/pierluigiferrari/ssd_keras

'''

from __future__ import division
import numpy as np
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

class DataGenerator:
    '''
    A generator to generate batches of images indefinitely.
    Can shuffle the dataset consistently after each complete pass.
    '''

    # todo : not possible to read hdf5 datasets, no eval_neutral, labels_output_format is fixed
    # only parser for csv files
    def __init__(self,
                 images_dir,
                 load_images_into_memory=False,
                 verbose=True,
                 show_images=False):
        '''

        In case you would like not to load any labels at all, simply pass a list of image filenames here.

        Arguments:
            images_dir (string): this should be the directory that contains the images
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
            show_images (bool, optional) : whether or not to visualize images, can be used to make sure the images used
            during the training are good
        '''

        self.load_images_into_memory = load_images_into_memory
        self.dataset_size = 0  # As long as we haven't loaded anything yet, the dataset size is zero.
        self.images = None
        self.show_images = show_images


        self.filenames = [os.path.join(images_dir, line.strip()) for line in os.listdir(images_dir)]

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

    def generate(self,
                 batch_size=32,
                 img_size = 32,
                 shuffle=True):
        '''
        Generates batches of samples

        Can shuffle the samples consistently after each complete pass.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            img_size : size of the images

        Yields:
            The next batch as the batch images, with the desired image size
        '''


        if self.dataset_size == 0:
            raise ValueError("Cannot generate batches because no images were found in the folder")

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.images is None):
                objects_to_shuffle.append(self.images)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]


        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X = []

            if current >= self.dataset_size:
                current = 0

            #########################################################################################
            # Maybe shuffle the dataset if a full pass over the dataset has finished.
            #########################################################################################

                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.images is None):
                        objects_to_shuffle.append(self.images)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]


            #########################################################################################
            # Get the images, image filenames
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, we'll have to load the individual image files from disk.
            batch_indices = self.dataset_indices[current:current + batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current + batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            current += batch_size



            #########################################################################################
            # Resize images to the desired shape
            #########################################################################################

            for i in range(len(batch_X)):
                batch_X[i] = cv2.resize(batch_X[i],
                                   dsize=(img_size, img_size),
                                   interpolation=cv2.INTER_LINEAR)

                ########################################################################################
                # visualize the batch items if needed
                ########################################################################################

                if self.show_images:
                    plt.figure(figsize=(12, 12))
                    plt.imshow(batch_X[i])

            batch_X = np.array(batch_X)

            if (batch_X.size == 0):
                raise ValueError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels.")


            yield batch_X


    def get_dataset(self):
        '''
        Returns:
            3-tuple containing lists and/or `None` for the filenames, labels, image IDs
        '''
        return self.filenames


    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return len(self.filenames)
