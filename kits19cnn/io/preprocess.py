import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join, isdir

class Preprocessor(object):
    """
    2D Preprocessing:
        * clipping (ROI)
            * allow ability to specify intensities
            * have ability to automatically do it
        * crop images to the nonint region (percentile_005)
        * mean_std_norm based on whole dset stats
            * allow ability to specify intensities
            * have ability to automatically do it
        * save as .npy array
            * imaging.npy
            * segmentation.npy
        * saves the crop coordinates in out_dir as "coords.csv"
    3D Preprocessing:
        2D Preprocessing but resample to median spacing beforehand
        Need to figure out how to uninterpolate
    """
    def __init__(self, in_dir, out_dir, clip_values=None, cases=None):
        """
        Attributes:
            in_dir (str): directory with the input data. Should be the kits19/data directory.
            out_dir (str): output directory where you want to save each case
            clip_values (list, tuple): values you want to clip CT scans to
                * For whole dset, the [0.5, 99.5] percentiles are [-75.75658734213053, 349.4891265535317]
            cases: list of case folders to preprocess
        """
        self.in_dir = in_dir
        self.out_dir = out_dir

        if clip_values is None:
            # automatically getting high/low values to clip to
            self.clip_values = self.get_clip_values()
        else:
            self.clip_values = clip_values

        self.cases = cases
        # automatically collecting all of the case folder names
        if self.cases is None:
            self.cases = [os.path.join(self.in_dir, case) \
                          for case in os.listdir(self.in_dir) \
                          if case.startswith("case")]
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
            preprocessed_img, preprocessed_label, coords = self.preprocess_2d(image, label)
            orig_shape = image.shape # need this for inference
            self.save_imgs(preprocessed_img, preprocessed_label, case)
            coords_dict = self.append_to_coords_dict(coords_dict, case, coords, orig_shape)
        df = pd.DataFrame(coords_dict)
        df.to_csv(join(self.out_dir, "coords.csv"))
        print("Done!")

    def preprocess_2d(self, image, mask):
        """
        Procedure:
        1) Clipping to specified values. Defaults to [0.5, 99.5 percentiles].
        2) Cropping out non-0.5-percentile region

        Args:
            image: numpy array
            mask: numpy array
        Returns:
            tuple(preprocessed (image, label), list of lists of coords)
        """
        clipped = np.clip(image, self.clip_values[0], self.clip_values[1])
        cropped_and_coords = extract_nonint_region(clipped, mask, outside_value=self.clip_values[0])
        return cropped_and_coords

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
        coords_dict["orig_z"].append(orig_z), coords_dict["orig_x"].append(orig_x), coords_dict["orig_y"].append(orig_y)

        return coords_dict

    def get_clip_values(self):
        """
        Automatically gathers the low/high values to clip to
        Returns:
            clip_values (tuple): [0.5, 99.5] percentiles of the ROI pixels to clip to
        """
        pixels = self.gather_roi_pixels()
        # The [0.5, 99.5] percentiles to clip to
        clip_values = (np.percentile(pixels, 0.5), np.percentile(pixels, 99.5))
        print("0.5 Percentile: {0}\n99.5 Percentile: {1}".format(percentiles[0], percentiles[1]))
        return clip_values

    def gather_roi_pixels(self):
        """
        Collects all of the segmentation ROI pixels in a numpy array.
        Returns:
            pixels (numpy array):
        """
        pixels = np.array([])
        for (i, case) in enumerate(self.cases):
            print("Processing {0}/{1}: {2}".format(i+1, len(self.cases), case))
            x = nib.load(os.path.join(self.in_dir, case, "imaging.nii.gz")).get_fdata()
            y = nib.load(os.path.join(self.in_dir, case, "segmentation.nii.gz")).get_fdata()
            overlap = x * y
            pixels = np.append(pixels, overlap[overlap != 0].flatten())
        return pixels

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
