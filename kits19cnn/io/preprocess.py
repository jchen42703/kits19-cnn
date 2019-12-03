import os
from os.path import join, isdir
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import nibabel as nib
import numpy as np
import json

from kits19cnn.io.resample import resample_patient

class Preprocessor(object):
    """
    Preprocesses the original dataset (interpolated).
    Procedures:
        * clipping (ROI)
        * save as .npy array
            * imaging.npy
            * segmentation.npy (if with_masks)
        * resampling from `orig_spacing` to `target_spacing`
            currently uses spacing reported in the #1 solution
    """
    def __init__(self, in_dir, out_dir, cases=None, kits_json_path=None,
                 target_spacing=(3.22, 1.62, 1.62),
                 clip_values=None, with_mask=False, fg_classes=[1, 2]):
        """
        Attributes:
            in_dir (str): directory with the input data. Should be the
                kits19/data directory.
            out_dir (str): output directory where you want to save each case
            cases: list of case folders to preprocess
            kits_json_path (str): path to the kits.json file in the kits19/data
                directory. This only should be specfied if you're resampling.
                Defaults to None.
            target_spacing (list/tuple): spacing to resample to
            clip_values (list, tuple): values you want to clip CT scans to.
                Defaults to None for no clipping.
            with_mask (bool): whether or not to preprocess with masks or no
                masks. Applicable to preprocessing test set (no labels
                available).
            fg_classes (list): of foreground class indices
        """
        self.in_dir = in_dir
        self.out_dir = out_dir

        self._load_kits_json(kits_json_path)
        self.clip_values = clip_values
        self.target_spacing = np.array(target_spacing)
        self.with_mask = with_mask
        self.fg_classes = fg_classes
        self.cases = cases
        # automatically collecting all of the case folder names
        if self.cases is None:
            self.cases = [os.path.join(self.in_dir, case) \
                          for case in os.listdir(self.in_dir) \
                          if case.startswith("case")]
            self.cases = sorted(self.cases)
            assert len(self.cases) > 0, \
                "Please make sure that in_dir refers to the proper directory."
        # making directory if out_dir doesn't exist
        if not isdir(out_dir):
            os.mkdir(out_dir)
            print("Created directory: {0}".format(out_dir))

    def gen_data(self):
        """
        Generates and saves preprocessed data
        Args:
            task_path: file path to the task directory (must have the corresponding "dataset.json" in it)
        Returns:
            preprocessed input image and mask
        """
        # Generating data and saving them recursively
        for case in tqdm(self.cases):
            x_path, y_path = join(case, "imaging.nii.gz"), join(case, "segmentation.nii.gz")
            image = nib.load(x_path).get_fdata()[None]
            label = nib.load(y_path).get_fdata()[None] if self.with_mask \
                    else None
            preprocessed_img, preprocessed_label = self.preprocess(image,
                                                                   label,
                                                                   case)

            self.save_imgs(preprocessed_img, preprocessed_label, case)

    def preprocess(self, image, mask, case=None):
        """
        Clipping, cropping, and resampling.
        Args:
            image: numpy array
            mask: numpy array or None
            case (str): path to a case folder
        Returns:
            tuple of:
                - preprocessed image
                - preprocessed mask or None
        """
        raw_case = Path(case).name # raw case name, i.e. case_00000
        if self.target_spacing is not None:
            for info_dict in self.kits_json:
                # guaranteeing that the info is corresponding to the right
                # case
                if info_dict["case_id"] == raw_case:
                    case_info_dict = info_dict
                    break
            orig_spacing = (case_info_dict["captured_slice_thickness"],
                            case_info_dict["captured_pixel_width"],
                            case_info_dict["captured_pixel_width"])
            image, mask = resample_patient(image, mask, np.array(orig_spacing),
                                           target_spacing=self.target_spacing)
        if self.clip_values is not None:
            image = np.clip(image, self.clip_values[0], self.clip_values[1])

        mask = mask[None] if mask is not None else mask
        return (image[None], mask)

    def save_imgs(self, image, mask, case):
        """
        Saves an image and mask pair as .npy arrays in the KiTS19 file structure
        Args:
            image: numpy array
            mask: numpy array
            case: path to a case folder (each element of self.cases)
        """
        # saving the generated dataset
        # output dir in KiTS19 format
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        np.save(os.path.join(out_case_dir, "imaging.npy"), image)
        if mask is not None:
            np.save(os.path.join(out_case_dir, "segmentation.npy"), mask)

    def save_dir_as_2d(self):
        """
        Takes preprocessed 3D numpy arrays and saves them as slices
        in the same directory.
        """
        self.pos_slice_dict = {}
        # Generating data and saving them recursively
        for case in tqdm(self.cases):
            # assumes the .npy files have shape: (n_channels, d, h, w)
            image = np.load(join(case, "imaging.npy"))
            label = np.load(join(case, "segmentation.npy"))
            image = image.squeeze(axis=0) if len(image.shape)==5 else image
            label = label.squeeze(axis=0) if len(label.shape)==5 else label

            self.save_3d_as_2d(image, label, case)
        self._save_pos_slice_dict()

    def save_3d_as_2d(self, image, mask, case):
        """
        Saves an image and mask pair as .npy arrays in the
        KiTS19 file structure
        Args:
            image: numpy array
            mask: numpy array
            case: path to a case folder (each element of self.cases)
        """
        # saving the generated dataset
        # output dir in KiTS19 format
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        # iterates through all slices and saves them individually as 2D arrays
        fg_indices = defaultdict(list)
        if mask.shape[1] <= 1:
            print("WARNING: Please double check your mask shape;",
                  f"Masks have shape {mask.shape} when it should be",
                  "shape (n_channels, d, h, w)")
            raise Exception("Please fix shapes.")
        for slice_idx in range(mask.shape[1]):
            label_slice = mask[:, slice_idx]
            # appending fg slice indices
            for idx in self.fg_classes:
                if (label_slice == idx).any():
                    fg_indices[idx].append(slice_idx)
            # naming convention: {type of slice}_{case}_{slice_idx}
            slice_idx_str = str(slice_idx)
            # adding 0s to slice_idx until it reaches 3 digits,
            # so sorting files is easier when stacking
            while len(slice_idx_str) < 3:
                slice_idx_str = "0"+slice_idx_str
            np.save(join(out_case_dir, f"imaging_{slice_idx_str}.npy"),
                    image[:, slice_idx])
            np.save(join(out_case_dir, f"segmentation_{slice_idx_str}.npy"),
                    label_slice)
        # {case1: [idx1, idx2,...], case2: ...}
        self.pos_slice_dict[case] = fg_indices

    def _save_pos_slice_dict(self):
        """
        Saves the foreground (positive) class dictionaries:
            - slice_indices.json
                saves the slice indices per class
                    {
                        case: {fg_class1: [slice indices...],
                               fg_class2: [slice indices...],
                               ...}
                    }
            - slice_indices_general.json
                saves the slice indices for all foreground classes into a
                    single list
                    {case: [slice indices...],}
        """
        # converting pos_slice_dict to general_slice_dict
        general_slice_dict = defaultdict(list)
        for case, slice_idx_dict in self.pos_slice_dict.items():
            for slice_idx_list in list(slice_idx_dict.values()):
                for slice_idx in slice_idx_list:
                    general_slice_dict[case].append(slice_idx)

        save_path = join(self.out_dir, "slice_indices.json")
        save_path_general = join(self.out_dir, "slice_indices_general.json")
        # saving the dictionaries
        print(f"Logged the slice indices for each class in {self.fg_classes} at"
              f"{save_path}.")
        with open(save_path, "w") as fp:
            json.dump(self.pos_slice_dict, fp)
        print("Logged slice indices for all fg classes instead of for each",
              f"class separately at {save_path_general}.")
        with open(save_path_general, "w") as fp:
            json.dump(general_slice_dict, fp)

    def _load_kits_json(self, json_path):
        """
        Loads the kits.json file into `self.kits_json`
        """
        if json_path is None:
            print("`kits_json_path is empty, so not resampling.`")
        elif json_path is not None:
            with open(json_path, "r") as fp:
                self.kits_json = json.load(fp)
