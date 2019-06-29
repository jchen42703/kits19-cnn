import numpy as np
import nibabel as nib
import os
from kits19cnn.io.gen_utils import BaseTransformGenerator

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
        label_probs (list, tuple): sequence of probabilities for each nonzero label. Default is None,
            which means classes are sampled uniformly.
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, fpaths, batch_size=2, n_pos=1, pos_slice_dict=None, label_probs=None,
                 transform=None, steps_per_epoch=None, shuffle=True):

        BaseTransformGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size, transform=transform,
                                        steps_per_epoch=steps_per_epoch, shuffle=shuffle)
        self.label_probs = label_probs
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
            # loads data as a numpy arr and then adds the channels dimension (channels_first)
            try:
                x_train = np.load(os.path.join(case_id, "imaging.npy"))[None]
                y_train = np.load(os.path.join(case_id, "segmentation.npy"))[None]
            except IOError:
                x_train = nib.load(os.path.join(case_id, "imaging.nii.gz")).get_fdata()[None]
                y_train = nib.load(os.path.join(case_id, "segmentation.nii.gz")).get_fdata()[None]
            # extracting slice:
            if pos_sample:
                if self.pos_slice_dict is None:
                    # did y_train[None] to add the batch size dimension
                    slice_idx = self.get_random_pos_slice_idx(y_train[None])
                else:
                    slice_idx = self.sample_pos_idx_from_dict(case_id)
            elif not pos_sample:
                # added batch size dimenion
                slice_idx = self.get_rand_slice_idx((1,)+x_train.shape)
            images_x.append(x_train[:, slice_idx]), images_y.append(y_train[:, slice_idx])
        return (images_x, images_y)

    def sample_pos_idx_from_dict(self, case_id):
        """
        Sampling a positive class slice index from the nested class label dictionary
        based on the user-defined label_probs attribute.
        Args:
            case_id (str): key of self.pos_slice_dict
        Returns
            slice_idx: the sampled scalar slice index
        """
        # sampling a positive class slice index from the nested class idx dictionary
        # based on the provided label_probs. if label_probs=None, the distrib is uniform.
        idx_dict = self.pos_slice_dict[case_id]
        # sampling the class
        class_labels = sorted(list(idx_dict.keys()))
        sample_class = np.random.choice(class_labels, p=self.label_probs)
        # getting a random slice index from the same sampled class
        slice_idx = np.random.choice(idx_dict[sample_class])
        return slice_idx

    def get_all_pos_slice_idx(self):
        """
        Done in the generator initialization if specified. Iterates through all labels and generates
        a dictionary of nested dictionaries {fname: {1: pos_idx1, 2: pos_idx2} pairs, where pos_idx
        is a tuple of all slice indices for each of the said case's labels.
        """
        pos_slice_dict = {}
        for (idx, case_id) in enumerate(self.fpaths):
            total = len(self.fpaths)
            print("Progress: {0}/{1}\nProcessing: {2}".format(idx+1, total, case_id))
            # loading the data iteratively;  [None][None] just adds on the
            # batch_size (1), and the n_channels (1) to make the shape
            # (1, 1, x, y, z); same as np.expand_dims(arr, 0) twice
            try:
                # support for .npy
                y_train = np.load(os.path.join(case_id, "segmentation.npy"))[None][None]
            except IOError:
                # support for .nii.gz
                y_train = nib.load(os.path.join(case_id, "segmentation.nii.gz")).get_fdata()[None][None]

            pos_slice_dict[case_id] = self.get_all_per_label_pos_slice_idx(y_train)
        return pos_slice_dict

    def get_all_per_label_pos_slice_idx(self, label):
        """
        Gets a all positive class slice indices for each class in the provided label
        (each index does not include the channels and batch_size dimensions.)
        Args:
            label: numpy array with the dims (batch, n_channels, x,y,z)
        Returns:
            dictionary with key, value pairs (class label (int), slice indices (list of int))
        """
        label_indices = np.unique(label)[1:] # excluding outside class
        n_labels = label_indices.size
        # dictionary nested under the eventually case_id with classes
        nested_class_dict = {class_label: np.where(label == class_label)[2].tolist() \
                             for class_label in label_indices}
        return nested_class_dict

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
