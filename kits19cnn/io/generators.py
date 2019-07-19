import numpy as np
import nibabel as nib
import os
import json
from kits19cnn.io.gen_utils import BaseTransformGenerator
from pathlib import Path

class SliceGenerator(BaseTransformGenerator):
    """
    Loads data, slices them based on the number of positive slice indices and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_first
    * .nii files should not have the batch_size dimension
    Attributes:
        fpaths: list of filenames
        batch_size: The number of images you want in a single batch
        n_pos: The number of positive class 2D images to include in a batch
        pos_slice_dict:
            (None): if you want to automatically get the dictionary of positive class slice indices
            (dict): if you want to manually provide it
            (anything else): if you want to find the slices on the fly
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, fpaths, batch_size=2, n_pos=1, pos_slice_dict=None, transform=None,
                 steps_per_epoch=None, shuffle=True):

        BaseTransformGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size, transform=transform,
                                        steps_per_epoch=steps_per_epoch, shuffle=shuffle)
        # handling different cases with positive class slicing;
        # getting it automatically, manually provided, or on the fly
        if pos_slice_dict is None:
            self.pos_slice_dict = self.get_all_pos_slice_idx()
        elif isinstance(pos_slice_dict, dict):
            self.pos_slice_dict = pos_slice_dict
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
            X, Y = self.shuffle_list(X, Y)
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
            try:
                x_train = np.expand_dims(np.load(os.path.join(case_id, "imaging.npy")), 0)
                y_train = np.expand_dims(np.load(os.path.join(case_id, "segmentation.npy")), 0)
            except IOError:
                x_train = np.expand_dims(nib.load(os.path.join(case_id, "imaging.nii.gz")).get_fdata(), 0)
                y_train = np.expand_dims(nib.load(os.path.join(case_id, "segmentation.nii.gz")).get_fdata(), 0)
            # extracting slice:
            if pos_sample:
                if self.pos_slice_dict is None:
                    slice_idx = self.get_rand_pos_slice_idx(np.expand_dims(y_train, 0))
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
            try:
                y_train = np.expand_dims(np.expand_dims(np.load(os.path.join(case_id, "segmentation.npy")), 0), 0)
            except IOError:
                y_train = np.expand_dims(np.expand_dims(nib.load(os.path.join(case_id, "segmentation.nii.gz")).get_fdata(), 0), 0)

            pos_slice_dict[case_id] = self.get_all_per_label_pos_slice_idx(y_train)
        return pos_slice_dict

    def get_all_per_label_pos_slice_idx(self, label):
        """
        Gets a random positive slice index. Assumes the background class is 0.
        Args:
            label: numpy array with the dims (batch, n_channels, x,y,z)
        Returns:
            a list of all non-background class integer slice indices
        """
        pos_idx = np.nonzero(label)[2]
        return pos_idx.squeeze().tolist()

    def get_rand_pos_slice_idx(self, label):
        """
        Gets a random positive slice index. Assumes the background class is 0.
        Args:
            label: numpy array with the dims (batch, n_channels, x,y,z)
        Returns:
            an integer representing a random non-background class slice index
        """
        # "n_dims" numpy arrays of all possible positive pixel indices for the label
        slice_indices = np.nonzero(label)[2]
        # finding random positive class index
        random_pos_coord = np.random.choice(slice_indices)
        return random_pos_coord

    def get_rand_slice_idx(self, shape):
        """
        Args:
            shape: data shape (includes the batch size and channels dims)
        Returns:
            A randomly selected slice index
        """
        return np.random.randint(0, shape[2]-1)

    def shuffle_list(self, *ls):
        """
        Shuffles lists together in the same way so pairs stay as pairs
        Args:
            *ls: list arguments
        Returns:
            Corresponding shuffled lists
        """
        l = list(zip(*ls))
        np.random.shuffle(l)
        return zip(*l)

class BinarySliceGenerator(BaseTransformGenerator):
    """
    Loads data, slices them based on the number of positive slice indices and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_first
    * Assumes that input data are composed of 2D numpy arrays, as produced by io.preprocess.Preprocessor when save_as_slices=True.

    TEMPORARY PLAN:
        Two OPTIONS:
        1) fpaths -> fpaths to case folders, sample using the results from listdir (conditions: .npy, stem ends with integers)
        2) fpaths -> fpaths to each individual slice and just use SMOTE to oversample filenames.
        Going with option 1 for now because it is the most compatible with the current pipeline.

    Attributes:
        fpaths: list of filenames
        pos_slice_dict_or_path:
            (str): path to the .json file for pos_slice_dict, generated by io.preprocess.Preprocessor when save_as_slices=True
            (dict): if you want to manually provide it
            (anything else): if you want to find the slices on the fly
        batch_size: The number of images you want in a single batch
        n_pos: The number of positive class 2D images to include in a batch
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, fpaths, pos_slice_dict_or_path, batch_size=2, n_pos=1, use_deep_supervision=True,
                 transform=None, steps_per_epoch=None, shuffle=True):

        BaseTransformGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size, transform=transform,
                                        steps_per_epoch=steps_per_epoch, shuffle=shuffle)
        if isinstance(pos_slice_dict_or_path, dict):
            self.pos_slice_dict = pos_slice_dict_or_path
        elif isinstance(pos_slice_dict_or_path, str):
            with open(pos_slice_dict_or_path, "r") as fp:
                self.pos_slice_dict = json.load(fp)

        self.use_deep_supervision = use_deep_supervision
        # handling different cases with positive class slicing:
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
            X, Y = self.shuffle_list(X, Y)
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
        # specifically for the attn_reg attention-gated u-net
        # scales the labels down
        if self.use_deep_supervision:
            Y1 = Y[:,:,::8,::8]
            Y2 = Y[:,:,::4,::4]
            Y3 = Y[:,:,::2,::2]
            Y4 = Y
            Y = [Y1,Y2,Y3,Y4]
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
            case_raw = Path(case_id).name
            # extracting slice:
            if pos_sample:
                slice_idx = self.get_rand_pos_slice_idx(case_raw)
            elif not pos_sample:
                slice_idx = self.get_rand_slice_idx(case_id)
            # formatting string
            slice_idx_str = str(slice_idx)
            while len(slice_idx_str) < 3:
                slice_idx_str = "0"+slice_idx_str
            # loads data as a numpy arr and then adds the channel + batch size dimensions
            x_train = np.expand_dims(np.load(os.path.join(case_id, "imaging_{0}.npy".format(slice_idx_str))), 0)
            y_train = np.expand_dims(np.load(os.path.join(case_id, "segmentation_{0}.npy".format(slice_idx_str))), 0)
            y_train[y_train == 1] = 0
            y_train[y_train == 2] = 1
            images_x.append(x_train), images_y.append(y_train)
        return (images_x, images_y)

    def get_rand_pos_slice_idx(self, case_raw):
        """
        Gets a random positive slice index from self.pos_slice_dict (that was generated by
        io.preprocess.Preprocessor when save_as_slices=True).
        Args:
            case_raw (str): raw case folder name (not the file path to it)
        Returns:
            an integer representing a random non-background class slice index
        """
        # finding random positive class index
        random_pos_coord = np.random.choice(self.pos_slice_dict[case_raw])
        return random_pos_coord

    def get_rand_slice_idx(self, case_fpath):
        """
        Args:
            case_fpath: each element of self.fpaths
        Returns:
            A randomly selected slice index
        """
        # assumes that there are no other files in said directory with "imaging_"
        _slice_files = [file for file in os.listdir(case_fpath) if file.startswith("imaging_")]
        return np.random.randint(0, len(_slice_files))

    def shuffle_list(self, *ls):
        """
        Shuffles lists together in the same way so pairs stay as pairs
        Args:
            *ls: list arguments
        Returns:
            Corresponding shuffled lists
        """
        l = list(zip(*ls))
        np.random.shuffle(l)
        return zip(*l)
