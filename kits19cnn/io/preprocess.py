import os
import nibabel as nib
from os.path import join, isdir
import numpy as np

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
            self.clip_values = self.get_clip_values()
        else:
            self.clip_values = clip_values

        self.cases = cases
        if cases is None:
            self.cases = [case for case in os.listdir(self.in_dir) \
                          if case.startswith("case")]
            assert len(self.cases) > 0, "Please make sure that in_dir refers to the kits19/data directory."
        # making directory if out_dir doesn't exist
        if not isdir(out_dir):
            os.mkdir(out_dir)
            print("Created directory: {0}".format(out_dir))

    def get_clip_values(self):
        pixels = gather_roi_pixels()
        # clipping to the [0.5, 99.5] percentiles
        self.clip_values = (np.percentile(pixels, 0.5), np.percentile(pixels, 99.5))
        print("0.5 Percentile: {0}\n99.5 Percentile: {1}".format(percentiles[0], percentiles[1]))

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

    def gen_data(self):
        """
        Generates and saves preprocessed data
        Args:
            task_path: file path to the task directory (must have the corresponding "dataset.json" in it)
        Returns:
            preprocessed input image and mask
        """
        # Generating data and saving them recursively
        for case in self.cases:
            image = nib.load(join(self.in_dir, case, "imaging.nii.gz")).get_fdata()
            label = nib.load(join(self.in_dir, case, "segmentation.nii.gz")).get_fdata()
            preprocessed_img, preprocessed_label = self.preprocess_2d(image, label, coords=False)
            self.save_imgs(preprocessed_img, preprocessed_label, case)
        print("Done!")

    def save_imgs(self, image, mask, case, pred=False):
        """
        Saves an image and mask pair as .npy arrays in the MSD file structure
        Args:
            image: numpy array
            mask: numpy array
            case: case folder name (each element of self.cases)
            pred (boolean): whether or not saving a prediction or preprocessed image
        """
        # saving the generated dataset
        # output dir in KiTS19 format
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)
            print("Created directory: {0}".format(out_images_dir))
        if pred:
            save_name = "pred_{0}.npy".format(case)
            np.save(os.path.join(out_case_dir, save_name), image)
            print("Saving prediction: {0}".format(save_name))
        else:
            np.save(os.path.join(out_case_dir, "imaging.npy"), image)
            np.save(os.path.join(out_case_dir, "segmentation.npy"), mask)
            print("Saving: {0}".format(case))

    def preprocess_2d(self, image, mask, coords=False):
        """
        Procedure:
        1) Clipping to specified values. Defaults to [0.5, 99.5 percentiles].
        2) Cropping out non-0.5-percentile region

        Args:
            image: numpy array
            mask: numpy array
            coords
        Returns:
            preprocessed (image, label) tuple
        """
        clipped = np.clip(image, self.clip_values[0], self.clip_values[1])
        cropped = extract_nonint_region(clipped, mask, outside_value=self.clip_values[0], coords=coords)
        return cropped

def extract_nonint_region(image, mask=None, outside_value=0, coords=False):
    """
    Resizing image around a specified region (i.e. nonzero region)
    Args:
        image: shape (x, y, z)
        mask: a segmentation labeled mask that is the same shaped as 'image' (optional; default: None)
        outside_value: (optional; default: 0)
        coords: boolean on whether or not to return boundaries (bounding box coords) (optional; default: False)
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
    if coords:
        if mask is not None:
            return (image[resizer], mask[resizer], coord_list)
        elif mask is None:
            return (image[resizer], coord_list)
    # returns just cropped outputs
    elif not coords:
        if mask is not None:
            return (image[resizer], mask[resizer])
        elif mask is None:
            return (image[resizer])
