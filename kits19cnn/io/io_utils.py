from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose
# from batchgenerators.transforms.crop_and_pad_transforms import PositiveClassCropTransform

def get_transforms(patch_shape = (80, 192, 128), pad_value = None, other_transforms = None, random_crop = False):
    """
    Initializes the transforms for training.
    Args:
        patch_shape:
        pad_value:
        other_transforms: List of transforms that you would like to add (optional). Defaults to None.
        random_crop (boolean): whether or not you want to random crop or center crop. Currently, the Transformed3DGenerator
        only supports random cropping. Transformed2DGenerator supports both random_crop = True and False.
    """
    # allowing for "nearest" padding
    if pad_value is None:
      border_mode = "nearest"
      pad_value = 0
    else:
      border_mode = "constant"
      border_cval_data = pad_value

    ndim = len(patch_shape)
    # brightness_transform = ContrastAugmentationTransform((0.3, 3.), preserve_range=True)
    spatial_transform = SpatialTransform(patch_shape,
                     do_elastic_deform = True, alpha = (0., 1500.), sigma = (30., 50.),
                     do_rotation = True, angle_z=(0, 2 * np.pi),
                     do_scale = True, scale = (0.8, 2.),
                     border_mode_data = border_mode, border_cval_data = pad_value,
                     order_data = 1, random_crop = random_crop,
                     p_el_per_sample=0.1, p_scale_per_sample=0.1, p_rot_per_sample=0.1)
    if ndim == 2:
        axes = (0, 1)
    elif ndim == 3:
        axes = (0, 1, 2)
    mirror_transform = MirrorTransform(axes = axes)
    transforms_list = [spatial_transform, mirror_transform]
    if other_transforms is not None:
        transforms_list = transforms_list + other_transforms
    composed = Compose(transforms_list)
    return composed

def get_only_crop(patch_shape = (512, 512), random_crop = False):
    """Returns a transform for only cropping"""
    spatial_transform = SpatialTransform(patch_shape,
                   do_elastic_deform = False,
                   do_rotation = False,
                   do_scale = False,
                   border_mode_data = "nearest", border_cval_data = 0,
                   order_data = 1, random_crop = random_crop)
    return spatial_transform
