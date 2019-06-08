import numpy as np
import nibabel as nib
import os
from kits19cnn.io.gen_utils import BaseTransformGenerator

class Slim3DGenerator(BaseTransformGenerator):
    """
    Depends on `batchgenerators.transforms` for the cropping and data augmentation.
    * Supports channels_first
    * .nii files should not have the batch_size dimension
    Attributes:
        fpaths: list of case folder names
        batch_size: The number of images you want in a single batch
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, fpaths, batch_size, transform=None, steps_per_epoch=1000, shuffle=True):

        BaseTransformGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size,
                               transform=transform, steps_per_epoch=steps_per_epoch, shuffle=shuffle)

    def data_gen(self, fpaths_temp):
        """
        Generates a batch of data.
        Args:
            fpaths_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two lists of numpy arrays: x, y
        """
        images_x = []
        images_y = []
        for case_id in fpaths_temp:
            # loads data as a numpy arr and then adds the channel + batch size dimensions
            x_train = np.expand_dims(nib.load(os.path.join(case_id, "imaging.nii")).get_fdata(), 0)
            y_train = np.expand_dims(nib.load(os.path.join(case_id, "segmentation.nii")).get_fdata(), 0)
            x_train = np.clip(x_train, -200, 300)
            images_x.append(x_train), images_y.append(y_train)
        return (images_x, images_y)

class SliceGenerator(BaseTransformGenerator):
    """
    Loads data, slices them based on the number of positive slice indices and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_first
    * .nii files should not have the batch_size dimension
    Attributes:
        fpaths: list of filenames
        batch_size: The number of images you want in a single batch
        n_pos: The number of positive class 2D images to include in a batch
        get_pos_dict (boolean): whether or not to get the dictionary of positive class slice indices
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, fpaths, batch_size=2, n_pos=1, get_pos_dict=False, transform=None,
                 steps_per_epoch=None, shuffle=True):

        BaseTransformGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size, transform=transform,
                                        steps_per_epoch=steps_per_epoch, shuffle=shuffle)
        if get_pos_dict:
            self.pos_slice_dict = self.get_all_pos_slice_idx()
        else:
            self.pos_slice_dict = None
        self.n_pos = n_pos
        if n_pos == 0:
            print("WARNING! Your data is going to be randomly sliced.")
            self.mode = "rand"
        elif n_pos == batch_size:
            print("WARNING! Your entire batch is going to be positively sampled.")
            self.mode = "pos"
        else:
            self.mode = "bal"

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Args:
            idx: the id assigned to each worker
        Returns:
        if self.pos_mask is True:
            (X,Y): a batch of transformed data/labels based on the n_pos attribute.
        elif self.pos_mask is False:
            ([X, Y], [Y, pos_mask]): multi-inputs for the capsule network decoder
        """
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        fpaths_temp = [self.fpaths[k] for k in indexes]
        # balanced sampling
        if self.mode == "bal":
            # generating data for both positive and randomly sampled data
            X_pos, Y_pos = self.data_gen(fpaths_temp[:self.n_pos], pos_sample=True)
            X_rand, Y_rand = self.data_gen(fpaths_temp[self.n_pos:], pos_sample=False)
            # concatenating all the corresponding data
            X, Y = X_pos+X_rand, Y_pos+Y_rand
            # shuffling the order of the positive/random patches
            np.random.shuffle(X), np.random.shuffle(Y)
        # random sampling
        elif self.mode == "rand":
            X, Y = self.data_gen(fpaths_temp, pos_sample=False)
        elif self.mode == "pos":
            X, Y = self.data_gen(fpaths_temp, pos_sample=True)
        # data augmentation
        if self.transform is not None:
            data_dict = {}
            data_dict["data"], data_dict["seg"] = X, Y
            data_dict = self.transform(**data_dict)
            X, Y = data_dict["data"], data_dict["seg"]
        return (X, Y)

    def data_gen(self, fpaths_temp, pos_sample):
        """
        Generates a batch of data.
        Args:
            fpaths_temp: batched list IDs; usually done by __getitem__
            pos_sample (boolean): if you want to sample an image with a nonzero class or not
        Returns:
            tuple of two numpy arrays: x, y
        """
        images_x = []
        images_y = []
        for case_id in fpaths_temp:
            # loads data as a numpy arr and then adds the channel + batch size dimensions
            x_train = np.expand_dims(nib.load(os.path.join(case_id, "imaging.nii")).get_fdata(), 0)
            y_train = np.expand_dims(nib.load(os.path.join(case_id, "segmentation.nii")).get_fdata(), 0)
            x_train = np.clip(x_train, -200, 300)
            # extracting slice:
            if pos_sample:
                if self.pos_slice_dict is None:
                    slice_idx = self.get_pos_slice_idx(np.expand_dims(y_train, 0))
                else:
                    slice_idx = np.random.choice(self.pos_slice_dict[case_id])
            elif not pos_sample:
                slice_idx = self.get_rand_slice_idx((1,)+x_train.shape)
            images_x.append(x_train[:, slice_idx]), images_y.append(y_train[:, slice_idx])
        return (images_x, images_y)

    def get_all_pos_slice_idx(self):
        """
        Done in the generator initialization if specified. Iterates through all labels and generates
        a dictionary of {fname: pos_idx} pairs, where pos_idx is a tuple of all positive class indices
        of said case's label.
        """
        pos_slice_dict = {}
        for (idx, case_id) in enumerate(self.fpaths):
            total = len(self.fpaths)
            print("Progress: {0}/{1}\nProcessing: {2}".format(idx+1, total, case_id))
            y_train = np.expand_dims(np.expand_dims(nib.load(os.path.join(case_id, "segmentation.nii")).get_fdata(), 0), 0)
            pos_slice_dict[case_id] = self.get_pos_slice_idx(y_train, return_all=True)
        return pos_slice_dict

    def get_rand_slice_idx(self, shape):
        """
        Args:
            shape: data shape (includes the batch size and channels dims)
        Returns:
            A randomly selected slice index
        """
        return np.random.randint(0, shape[2]-1)

    def get_pos_slice_idx(self, label, return_all=False):
        """
        Gets a random positive patch index that does not include the channels and batch_size dimensions.
        Args:
            label: one-hot encoded numpy array with the dims (batch, n_channels, x,y,z)
            return_all (boolean): whether or not to return the entirety of the array
        Returns:
            a list representing a 3D random positive patch index (same number of dimensions as label)
        """
        # "n_dims" numpy arrays of all possible positive pixel indices for the label
        pos_idx = np.nonzero(label)[2]
        if return_all:
            return tuple(pos_idx.squeeze())
        else:
            # finding random positive class index
            pos_idx = np.dstack(pos_idx).squeeze()
            random_coord_idx = np.random.choice(pos_idx.shape[0]) # choosing random coords out of pos_idx
            random_pos_coord = list(pos_idx[random_coord_idx])
            return random_pos_coord
