from batchgenerators.transforms import AbstractTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop
import numpy as np

from kits19cnn.io.custom_augmentations import foreground_crop

class ROICropTransform(AbstractTransform):
    """
    Crops the foreground in images `p_per_sample` part of the time. The
    fallback cropping is center cropping.
    """
    def __init__(self, crop_size=128, margins=(0, 0, 0), data_key="data",
                 label_key="seg", coords_key="bbox_coords",
                 p_per_sample=0.33):
        self.data_key = data_key
        self.label_key = label_key
        self.coords_key = coords_key
        self.margins = margins
        self.crop_size = crop_size
        self.p_per_sample = p_per_sample
        print(f"Cropping the foreground with a probabily: {p_per_sample}")

    def __call__(self, **data_dict):
        """
        Actually doing the cropping. Make sure that data_dict has the
        a key for the coords (self.coords_key) if p >0.
        (If the output of data_dict.get(self.coords_key) is None, then foreground
        crops are done on-the-fly).
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        if np.random.uniform() < self.p_per_sample:
            coords = data_dict.get(self.coords_key)

            data, seg = foreground_crop(data, seg, patch_size=self.crop_size,
                                        margins=self.margins,
                                        bbox_coords=coords)
        else:
            data, seg = center_crop(data, self.crop_size, seg)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
