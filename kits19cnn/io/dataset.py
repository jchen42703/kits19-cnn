from os.path import isfile, join
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset

class VoxelDataset(Dataset):
    def __init__(self, im_ids: np.array,
                 transforms=None,
                 preprocessing=None,
                 file_ending: str = ".npy"):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
        """
        self.im_ids = im_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.file_ending = file_ending
        print(f"Using the {file_ending} files...")

    def __getitem__(self, idx):
        # loads data as a numpy arr and then adds the channel + batch size dimensions
        case_id = self.im_ids[idx]
        x, y = self.load_volume(case_id)
        if self.transforms:
            x = x[None] if len(x.shape) == 4 else x
            y = y[None] if len(y.shape) == 4 else y
            # batchgenerators requires shape: (b, c, ...)
            data_dict = self.transforms(**{"data": x, "seg": y})
            x, y = data_dict["data"], data_dict["seg"]
        if self.preprocessing:
            preprocessed = self.preprocessing(**{"data": x, "seg": y})
            x, y = preprocessed["data"], preprocessed["seg"]
        # squeeze to remove batch size dim
        x = torch.squeeze(x, dim=0).float()
        y = torch.squeeze(y, dim=0)
        return (x, y)

    def __len__(self):
        return len(self.im_ids)

    def load_volume(self, case_id):
        """
        Loads volume from either .npy or nifti files.
        Args:
            case_id: path to the case folder
                i.e. /content/kits19/data/case_00001
        Returns:
            Tuple of:
            - x (np.ndarray): shape (1, d, h, w)
            - y (np.ndarray): same shape as x
        """
        x_path = join(case_id, f"imaging{self.file_ending}")
        y_path = join(case_id, f"segmentation{self.file_ending}")
        if self.file_ending == ".npy":
            x, y = np.load(x_path), np.load(y_path)
        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            x, y = nib.load(x_path).get_fdata(), nib.load(y_path).get_fdata()
        return (x[None], y[None])

class ClfSegVoxelDataset(VoxelDataset):
    """
    Can handle classification+segmentation, classification only, and seg only
    outputs.
    """
    def __init__(self, im_ids: np.array,
                 transforms=None,
                 preprocessing=None,
                 file_ending: str = ".npy",
                 mode: str = "both",
                 num_classes: int = 3):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
            mode (str): decides how the nature of the outputs
                must be one of ['both', 'clf_only', 'seg_only']
            num_classes (int): number of classes. Defaults to 3.
        """
        super().__init__(im_ids=im_ids, transforms=transforms,
                         preprocessing=preprocessing, file_ending=file_ending)
        assert mode.lower() in ["both", "clf_only", "seg_only"], \
            "`mode` must be one of ['both', 'clf_only', 'seg_only']"
        self.mode = mode
        self.num_classes = num_classes

    def __getitem__(self, idx):
        # loads data as a numpy arr and then adds the channel + batch size dimensions
        case_id = self.im_ids[idx]
        x, y = self.load_volume(case_id)
        if self.transforms:
            x = x[None] if len(x.shape) == 4 else x
            y = y[None] if len(y.shape) == 4 else y
            # batchgenerators requires shape: (b, c, ...)
            data_dict = self.transforms(**{"data": x, "seg": y})
            x, y = data_dict["data"], data_dict["seg"]

        if self.mode in ["both", "clf_only"]:
            y_clf = torch.from_numpy(self.get_clf_label_from_cropped_mask(y))

        if self.preprocessing:
            preprocessed = self.preprocessing(**{"data": x, "seg": y})
            x, y = preprocessed["data"], preprocessed["seg"]
        # squeeze to remove batch size dim
        if torch.is_tensor(x):
            x = torch.squeeze(x, dim=0).float()
        if torch.is_tensor(y):
            y = torch.squeeze(y, dim=0)

        if self.mode == "both":
            return (x, (y, y_clf))
        elif self.mode == "clf_only":
            return (x, y_clf)
        elif self.mode == "seg_only":
            return (x, y)

    def get_clf_label_from_cropped_mask(self, cropped_mask: np.array):
        """
        Multi-label one-hot encoding of mask to get the classification
        label.
        Args:
            cropped_mask (np.ndarray): contains int in [0, num_classes-1]
        Returns:
            one_hot (np.ndarray): multi-label one hot encoded array
                i.e. [0, 1, 0] or [1, 0, 1], etc.
        """
        unique = np.unique(cropped_mask)
        one_hot = np.zeros(self.num_classes)
        one_hot[unique] = 1
        return one_hot

class TestVoxelDataset(VoxelDataset):
    """
    Same as VoxelDataset, but can handle when there are no masks (just returns
    blank masks). This is a separate class to prevent lowkey errors with
    blank masks--VoxelDataset explicitly fails when there are no masks.
    """
    def __init__(self, im_ids: np.array,
                 transforms=None,
                 preprocessing=None,
                 file_ending=".npy"):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
        """
        super().__init__(im_ids=im_ids, transforms=transforms,
                         preprocessing=preprocessing, file_ending=file_ending)

    def load_volume(self, case_id):
        """
        Loads volume from either .npy or nifti files.
        Args:
            case_id: path to the case folder
                i.e. /content/kits19/data/case_00001
        Returns:
            Tuple of:
            - x (np.ndarray): shape (1, d, h, w)
            - y (np.ndarray): same shape as x
                if this does not exist, it's returned as a blank mask
        """
        x_path = join(case_id, f"imaging{self.file_ending}")
        y_path = join(case_id, f"segmentation{self.file_ending}")
        if self.file_ending == ".npy":
            x = np.load(x_path)
            y = np.load(y_path) if isfile(y_path) else np.zeros(x.shape)
        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            x = nib.load(x_path).get_fdata()
            y = nib.load(y_path).get_fdata() if isfile(y_path) else np.zeros(x.shape)
        return (x[None], y[None])
