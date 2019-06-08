from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, ResizeTransform
from batchgenerators.transforms.utility_transforms import ConvertMultiSegToOnehotTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.abstract_transforms import Compose
import numpy as np

def get_transforms(patch_shape=(80, 192, 128), other_transforms=None, minimal=True, random_crop=False):
    """
    Initializes the transforms for training.
    Args:
        patch_shape:
        other_transforms: List of transforms that you would like to add (optional). Defaults to None.
        minimal (boolean): whether to include more in-depth data aug (random rotations, random elastic deformations,
        random crops, random zoom, mirroring) or not; True if no daug, False if yes more daug
        random_crop (boolean): whether or not you want to random crop or center crop.
    """
    resize =  ResizeTransform(patch_shape, concatenate_list=True)
    mean_std_norm = ZeroMeanUnitVarianceTransform()
    onehot = ConvertMultiSegToOnehotTransform([1, 2])

    ndim = len(patch_shape)
    spatial_transform = SpatialTransform(patch_shape,
                     do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                     do_rotation=True, angle_z=(0, 2 * np.pi),
                     do_scale=True, scale=(0.8, 2.),
                     border_mode_data="nearest",
                     order_data=3, random_crop=random_crop,
                     p_el_per_sample=0.1, p_scale_per_sample=0.1, p_rot_per_sample=0.1)
    if ndim == 2:
        axes = (0, 1)
    elif ndim == 3:
        axes = (0, 1, 2)
    mirror_transform = MirrorTransform(axes = axes)
    if minimal:
        transforms_list = [resize, onehot, mean_std_norm]
    else:
        transforms_list = [resize, onehot, mean_std_norm, spatial_transform, mirror_transform]
    if other_transforms is not None:
        transforms_list = transforms_list + other_transforms
    composed = Compose(transforms_list)
    return composed
