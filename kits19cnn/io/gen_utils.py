import tensorflow.keras as keras
import numpy as np
import os
import nibabel as nib

class BaseGenerator(keras.utils.Sequence):
    """
    Basic framework for generating thread-safe data in keras. (no preprocessing and channels_last)
    Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Attributes:
      fpaths: filenames (.nii files); must be same for training and labels
      batch_size: int of desired number images per epoch
      shuffle: boolean on whether or not to shuffle the dataset
    """
    def __init__(self, fpaths, batch_size, shuffle=True):
        # lists of paths to images
        self.fpaths = fpaths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.fpaths))

    def __len__(self):
        return int(np.ceil(len(self.fpaths) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        """
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Find list of IDs
        fpaths_temp = [self.fpaths[k] for k in indexes]

        X, Y = self.data_gen(fpaths_temp)
        return (X, Y)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # self.img_idx = np.arange(len(self.x))
        self.indexes = np.arange(len(self.fpaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_gen(self, fpaths_temp):
        """
        Preprocesses the data
        Args:
            fpaths_temp: temporary batched list of ids (filenames)
        Returns
            x, y
        """
        raise NotImplementedError

class BaseTransformGenerator(BaseGenerator):
    """
    Loads data and applies data augmentation with `batchgenerators.transforms`.
    Attributes:
        fpaths: list of filenames
        batch_size: The number of images you want in a single batch
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        steps_per_epoch: steps per epoch during training (number of samples per epoch = steps_per_epoch * batch_size )
        shuffle: boolean on whether to shuffle the dataset between epochs
    """
    def __init__(self, fpaths, batch_size, transform=None, steps_per_epoch=None, shuffle=True):

        BaseGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size,
                               shuffle=shuffle)
        self.transform = transform
        n_samples = len(self.fpaths)
        self.indexes = np.arange(n_samples)
        if steps_per_epoch is None:
            steps_per_epoch = n_samples
        n_idx = self.batch_size * steps_per_epoch # number of samples per epoch
        # Handles cases where the dataset is small and the batch size is high
        if n_idx > n_samples:
            print("Adjusting the indexes since the total number of required samples (steps_per_epoch * batch_size) is greater than",
            "the number of provided images.")
            self.adjust_indexes(n_idx)
            print("Done!")
        assert self.indexes.size == n_idx

    def adjust_indexes(self, n_idx):
        """
        Adjusts self.indexes to the length of n_idx.
        """
        assert n_idx > self.indexes.size, "WARNING! The n_idx should be larger than the current number of indexes or else \
                                           there's no point in using this function. It has been automatically adjusted for you."
        # expanding the indexes until it passes the threshold: max_n_idx (extra will be removed later)
        while n_idx > self.indexes.size:
            self.indexes = np.repeat(self.indexes, 2)
        remainder = (len(self.indexes) % (n_idx))
        if remainder != 0:
            self.indexes = self.indexes[:-remainder]

        try:
            assert n_idx == self.indexes.size, "Expected number of indices per epoch does not match self.indexes.size."
        except AssertionError:
            raise Exception("Please make your steps_per_epoch > 3 if your batch size is < 3.")

    def __len__(self):
        """
        Steps per epoch (total number of samples per epoch / batch size)
        """
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Returns a batch of data (x,y)
        """
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        fpaths_temp = [self.fpaths[k] for k in indexes]
        X, Y = self.data_gen(fpaths_temp)
        if self.transform is not None:
            data_dict = {}
            data_dict["data"], data_dict["seg"] = X, Y
            data_dict = self.transform(**data_dict)
            X, Y = data_dict["data"], data_dict["seg"]
        return (X, Y)

    def data_gen(self, fpaths_temp):
        """
        Generates a batch of data.
        Args:
            fpaths_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two numpy arrays: x, y
        """
        raise NotImplementedError
