import numpy as np
import tensorflow as tf
import cv2
import random

from map_image import MapImage


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generator of data batches, to be fed into the model during
    training and testing.
    """

    def __init__(self, image_ids,
                 map_id_to_path,
                 map_id_to_class,
                 batch_size=32,
                 output_dim=(224, 224, 3),
                 n_classes=10,
                 shuffle=True,
                 train=True,
                 debug=False):
        self.image_ids = image_ids
        self.map_id_to_path = map_id_to_path
        self.map_id_to_class = map_id_to_class

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.n_classes = n_classes

        self.shuffle = shuffle

        # TODO: floods memory
        # self.map_images = self._load_map_images()

        # flag to distinguish train from test mode
        self.train = train
        self.debug = debug

        self.indexes = None
        self.on_epoch_end()

    def _load_map_images(self):

        map_images = dict()
        for image_id in self.image_ids:
            map_image = MapImage(image_id=image_id,
                                 image_path=self.map_id_to_path[image_id])

            # TODO: enable them to make training faster
            #             map_image.preprocess_image_new()
            #             map_image.normalize_pixel_range()

            map_images[image_id] = map_image

        return map_images

    def _load_map_image(self, image_id):

        map_image = MapImage(image_id=image_id,
                             image_path=self.map_id_to_path[image_id])

        return map_image

    def __getitem__(self, index):
        """
        Returns one batch of data
        """
        # generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]

        # Find list of image ids
        list_ids_temp = [self.image_ids[k] for k in indexes]

        # generate data
        if self.debug:
            X, y, original_images = self._data_generation(list_ids_temp)
        else:
            X, y = self._data_generation(list_ids_temp)

        if self.train:
            if self.debug:
                return X, y, original_images
            else:
                return X, y
        else:
            return X

    def _data_generation(self, image_ids_batch):
        """
        Generates data containing batch_size samples
        """
        X = np.empty((self.batch_size, self.output_dim[0],
                      self.output_dim[1], self.output_dim[2]))
        y = np.empty(self.batch_size, dtype=int)

        # used for debugging purposes
        original_images = list()

        for i, image_id in enumerate(image_ids_batch):

            map_image = self._load_map_image(image_id)

            if self.debug:
                # store
                original_images.append(np.copy(map_image.image))

            map_image.preprocess_image_new()
            map_image.normalize_pixel_range()
            # self.map_images[image_id].plot()

            image = map_image.image
            # image = self._get_square_crop(image, 224)
            image = self._get_random_square_crop(image, self.output_dim[0])
            # plt.imshow(image)
            # plt.show()

            # store image in numpy array
            X[i, :] = image
            if self.train:
                y[i] = self.map_id_to_class[image_id]

        if self.debug:
            return X, y, original_images
        else:
            return X, y

    @staticmethod
    def _get_square_crop(image, length):
        """
        Returns a centered squared crop of maximal dimensions
        """
        #         initial_square_length = min(image.shape[0], image.shape[1])

        width = image.shape[1]
        height = image.shape[0]

        if width > height:
            # width > height
            square_length = height
            x0 = 0
            x1 = square_length
            y0 = int((width / 2) - (square_length / 2))
            y1 = int((width / 2) + (square_length / 2))

        else:
            # height >=  width
            square_length = width
            x0 = int((height / 2) - (square_length / 2))
            x1 = int((height / 2) + (square_length / 2))
            y0 = 0
            y1 = square_length

        image = image[x0:x1, y0:y1, :]

        final_dim = (length, length)
        image = cv2.resize(image, final_dim)

        return image

    @staticmethod
    def _get_random_square_crop(image, length):
        """
        Returns a centered squared crop of maximal dimensions
        """
        width = image.shape[1]
        height = image.shape[0]

        noise = random.uniform(-0.95, 0.95)
        # noise_2 = random.uniform(0.90, 1) # do not allow too low values

        if width > height:
            # width > height
            square_length = height
            x0 = 0
            x1 = square_length

            d = (width - square_length) / 2
            y0 = int(d + (noise * d))
            y1 = int(y0 + square_length)
            # y0 = int((width / 2) - (square_length / 2))
            # y1 = int((width / 2) + (square_length / 2))

        else:
            # height >=  width
            square_length = width
            y0 = 0
            y1 = square_length

            d = (height - square_length) / 2
            x0 = int(d + (noise * d))
            x1 = int(x0 + square_length)
            # x0 = int((height / 2) - (square_length / 2))
            # x1 = int((height / 2) + (square_length / 2))

        image = image[x0:x1, y0:y1, :]

        final_dim = (length, length)
        image = cv2.resize(image, final_dim)

        return image

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)