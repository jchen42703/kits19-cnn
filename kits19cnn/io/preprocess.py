import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join, isdir

from .resample import resample_patient

class Preprocessor(object):
    """
    Preprocessing:
        * clipping (ROI)
        * crop images to the nonint region (percentile_005)
        * save as .npy array
            * imaging.npy
            * segmentation.npy
        * saves the crop coordinates in out_dir as "coords.csv"
        * resampling from `orig_spacing` to `target_spacing`
    """
    def __init__(self, in_dir, out_dir, cases=None,
                 orig_spacing=(3, 0.78162497, 0.78162497),
                 target_spacing=(3.22, 1.62, 1.62),
                 clip_values=None, extract_nonint=False):
        """
        Attributes:
            in_dir (str): directory with the input data. Should be the kits19/data directory.
            out_dir (str): output directory where you want to save each case
            cases: list of case folders to preprocess
            orig_spacing (list/tuple): spacing of nifti files
                Assumes same spacing
            target_spacing (list/tuple): spacing to resample to
            clip_values (list, tuple): values you want to clip CT scans to.
                Defaults to None for no clipping.
            extract_nonint (bool): whether or not to extract the non-zero
                regions for preprocessing
        """
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.clip_values = clip_values

        self.orig_spacing = np.array(orig_spacing)
        self.target_spacing = np.array(target_spacing)
        self.extract_nonint = extract_nonint

        self.cases = cases
        # automatically collecting all of the case folder names
        if self.cases is None:
            self.cases = [os.path.join(self.in_dir, case) \
                          for case in os.listdir(self.in_dir) \
                          if case.startswith("case")]
            self.cases = sorted(self.cases)[:210]
            assert len(self.cases) > 0, "Please make sure that in_dir refers to the kits19/data directory."
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
        coords_dict = {"cases": [],
                      "z_lb": [], "z_ub": [],
                      "x_lb": [], "x_ub": [],
                      "y_lb": [], "y_ub": [],
                      "orig_z": [], "orig_x": [], "orig_y": [],
                      }
        # Generating data and saving them recursively
        for (i, case) in enumerate(self.cases):
            print("Processing {0}/{1}: {2}".format(i+1, len(self.cases), case))
            image = nib.load(join(case, "imaging.nii.gz")).get_fdata()
            label = nib.load(join(case, "segmentation.nii.gz")).get_fdata()
            preprocessed_img, preprocessed_label, coords = self.preprocess(image, label)
            orig_shape = image.shape # need this for inference
            self.save_imgs(preprocessed_img, preprocessed_label, case)
            coords_dict = self.append_to_coords_dict(coords_dict, case, coords, orig_shape)
        df = pd.DataFrame(coords_dict)
        df.to_csv(join(self.out_dir, "coords.csv"))
        print("Done!")

    def preprocess(self, image, mask):
        """
        Clipping, cropping, and resampling.
        Args:
            image: numpy array
            mask: numpy array
        Returns:
            tuple of:
                - preprocessed image
                - preprocessed mask
                - list of lists of coords
        """
        if self.target_spacing:
            image, mask = resample_patient(image, mask, self.orig_spacing
                                           target_spacing=self.target_spacing)
        if self.clip_values:
            image = np.clip(image, self.clip_values[0], self.clip_values[1])
        if self.extract_nonint:
            image, mask, coords = extract_nonint_region(image, mask,
                                                        outside_value=self.clip_values[0])
            return (image, mask, coords)
        else:
            return (image, mask)

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
            print("Created directory: {0}".format(out_case_dir))

        np.save(os.path.join(out_case_dir, "imaging.npy"), image)
        np.save(os.path.join(out_case_dir, "segmentation.npy"), mask)
        print("Saving: {0}".format(case))

    def append_to_coords_dict(self, coords_dict, case, coords, orig_shape):
        """
        Simple function to append the coords and cases to the coords_dict.
        Args:
            coords_dict: dictionary containing all of the coordinates
            case (str): case folder name
            coords: list of [lower bound, upper bound]
        Returns:
            new coords_dict with the append coordinates
        """
        # case comes in as a filepath, so let's just get the case id
        case = Path(case).name
        # unpacking coords (list of lists of [lower bound, upper bound])
        # lb = lower bound, ub = upper bound
        z_lb, z_ub = coords[0]
        x_lb, x_ub = coords[1]
        y_lb, y_ub = coords[2]
        orig_z, orig_x, orig_y = orig_shape
        coords_dict["cases"].append(case)
        coords_dict["z_lb"].append(z_lb), coords_dict["z_ub"].append(z_ub)
        coords_dict["x_lb"].append(x_lb), coords_dict["x_ub"].append(x_ub)
        coords_dict["y_lb"].append(y_lb), coords_dict["y_ub"].append(y_ub)
        coords_dict["orig_z"].append(orig_z)
        coords_dict["orig_x"].append(orig_x)
        coords_dict["orig_y"].append(orig_y)

        return coords_dict

def extract_nonint_region(image, mask=None, outside_value=0):
    """
    Resizing image around a specified region (i.e. nonzero region)
    Args:
        image: shape (x, y, z)
        mask: a segmentation labeled mask that is the same shaped as 'image' (optional; default: None)
        outside_value: (optional; default: 0)
    Returns:
        the resized image
        segmentation mask (when mask is not None)
        a nested list of the mins and and maxes of each axis (when coords = True)
    """
    # Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ===================================================================================================
    # Changes: Added the ability to return the cropping coordinates
    pos_idx = np.where(image != outside_value)
    # fetching all of the min/maxes for each axes
    pos_x, pos_y, pos_z = pos_idx[1], pos_idx[2], pos_idx[0]
    minZidx, maxZidx = int(np.min(pos_z)), int(np.max(pos_z)) + 1
    minXidx, maxXidx = int(np.min(pos_x)), int(np.max(pos_x)) + 1
    minYidx, maxYidx = int(np.min(pos_y)), int(np.max(pos_y)) + 1
    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    coord_list = [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]
    # returns cropped outputs with the bbox coordinates
    if mask is not None:
        return (image[resizer], mask[resizer], coord_list)
    elif mask is None:
        return (image[resizer], coord_list)
