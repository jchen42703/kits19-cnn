import numpy as np
import math
import random
from batchgenerators.transforms import AbstractTransform

from kits19cnn.io.custom_augmentations import foreground_crop, center_crop, \
                                              random_resized_crop

class RandomResizedCropTransform(AbstractTransform):
    """
    Crop the given array to random size and aspect ratio.
    Doesn't resize across the depth dimenion (assumes it is dim=0) if
    the data is 3D.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Assumes the data and segmentation masks are the same size.
    """
    def __init__(self, target_size, scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 data_key="data", label_key="seg", p_per_sample=0.33,
                 crop_kwargs={}, resize_kwargs={}):
        """
        Attributes:
            pass
        """
        if len(target_size) > 2:
            print("Currently only adjusts the aspect ratio for the 2D dims.")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        self.target_size = target_size
        self.scale = scale
        self.ratio = ratio
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.crop_kwargs = crop_kwargs
        self.resize_kwargs = resize_kwargs

    def _get_image_size(self, data):
        """
        Assumes data has shape (b, c, h, w (, d)). Fetches the h, w, and d.
        depth if applicable.
        """
        return data.shape[2:]

    def get_crop_size(self, data, scale, ratio):
        """
        Get parameters for ``crop`` for a random sized crop.
        """
        shape_dims = self._get_image_size(data)
        area = np.prod(shape_dims)

        while True:
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if len(shape_dims) == 3:
                depth = shape_dims[0]
                crop_size = np.array([depth, h, w])
            else:
                crop_size = np.array([h, w])

            if (crop_size <= shape_dims).all() and (crop_size > 0).all():
                return crop_size

    def __call__(self, **data_dict):
        """
        Actually doing the cropping.
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        if np.random.uniform() < self.p_per_sample:
            crop_size = self.get_crop_size(data, self.scale, self.ratio)
            data, seg = random_resized_crop(data, seg,
                                            target_size=self.target_size,
                                            crop_size=crop_size,
                                            crop_kwargs=self.crop_kwargs,
                                            resize_kwargs=self.resize_kwargs)
        else:
            data, seg = center_crop(data, self.target_size, seg,
                                    crop_kwargs=self.crop_kwargs)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg.astype(np.float32)

        return data_dict

class ROICropTransform(AbstractTransform):
    """
    Crops the foreground in images `p_per_sample` part of the time. The
    fallback cropping is center cropping.
    """
    def __init__(self, crop_size=128, margins=(0, 0, 0), data_key="data",
                 label_key="seg", coords_key="bbox_coords",
                 p_per_sample=0.33, crop_kwargs={}):
        self.data_key = data_key
        self.label_key = label_key
        self.coords_key = coords_key
        self.margins = margins
        self.crop_size = crop_size
        self.p_per_sample = p_per_sample
        self.crop_kwargs = crop_kwargs

    def __call__(self, **data_dict):
        """
        Actually doing the cropping. Make sure that data_dict has the
        a key for the coords (self.coords_key) if p>0.
        (If the output of data_dict.get(self.coords_key) is None, then foreground
        crops are done on-the-fly).
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        if np.random.uniform() < self.p_per_sample:
            coords = data_dict.get(self.coords_key)

            data, seg = foreground_crop(data, seg, patch_size=self.crop_size,
                                        margins=self.margins,
                                        bbox_coords=coords,
                                        crop_kwargs=self.crop_kwargs)
        else:
            data, seg = center_crop(data, self.crop_size, seg,
                                    crop_kwargs=self.crop_kwargs)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

class MultiClassToBinaryTransform(AbstractTransform):
    """
    For changing a multi-class case to a binary one. Specify the label to
    change to binary with `roi_label`.
    - Don't forget to adjust `remove_label` accordingly!
    - label will be turned to a binary label with only `roi_label`
      existing as 1s
    """
    def __init__(self, roi_label="2", remove_label="1", label_key="seg"):
        self.roi_label = int(roi_label)
        self.remove_label = int(remove_label)
        self.label_key = label_key

    def __call__(self, **data_dict):
        """
        Replaces the label values
        """
        label = data_dict.get(self.label_key)
        # changing labels
        label[label == self.remove_label] = 0
        label[label == self.roi_label] = 1

        data_dict[self.label_key] = label

        return data_dict

class RepeatChannelsTransform(AbstractTransform):
    """
    Repeats across the channels dimension `num_tiles` number of times.
    """
    def __init__(self, num_repeats=3, data_key="data"):
        self.num_repeats = num_repeats
        self.data_key = data_key

    def __call__(self, **data_dict):
        """
        Repeats across the channels dimension (axis=1).
        """
        data = data_dict.get(self.data_key)

        data_dict[self.data_key] = np.repeat(data, self.num_repeats, axis=1)

        return data_dict
