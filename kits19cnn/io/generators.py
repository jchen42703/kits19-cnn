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
        list_IDs: list of case folder names
        batch_size: The number of images you want in a single batch
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, list_IDs, batch_size, transform=None, steps_per_epoch=1000, shuffle=True):

        BaseTransformGenerator.__init__(self, list_IDs=list_IDs, batch_size=batch_size,
                               transform=transform, steps_per_epoch=steps_per_epoch, shuffle=shuffle)

    def data_gen(self, list_IDs_temp):
        """
        Generates a batch of data.
        Args:
            list_IDs_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two lists of numpy arrays: x, y
        """
        images_x = []
        images_y = []
        for case_id in list_IDs_temp:
            # loads data as a numpy arr and then adds the channel + batch size dimensions
            x_train = np.expand_dims(nib.load(os.path.join(case_id, "imaging.nii")).get_fdata(), 0)
            y_train = np.expand_dims(nib.load(os.path.join(case_id, "segmentation.nii")).get_fdata(), 0)
            x_train = np.clip(x_train, -200, 300)
            images_x.append(x_train), images_y.append(y_train)
        return (images_x, images_y)
