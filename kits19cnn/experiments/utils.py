import os
import random
import json
import numpy as np
import batchgenerators.transforms as bg
import torch
from copy import deepcopy

from kits19cnn.io import ROICropTransform, RepeatChannelsTransform

bgut = bg.utility_transforms
bgct = bg.color_transforms
bgsnt = bg.sample_normalization_transforms

def get_training_augmentation(augmentation_key="aug1"):
    default_angle = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    aug1_spatial_kwargs = {"patch_size": (80, 160, 160),
                           "patch_center_dist_from_border": (30, 30, 30),
                           "do_elastic_deform": True,
                           "alpha": (0., 900.),
                           "sigma": (9., 13.),
                           "do_rotation": True,
                           "angle_x": default_angle,
                           "angle_y": default_angle,
                           "angle_z": default_angle,
                           "do_scale": True,
                           "scale": (0.85, 1.25),
                           "border_mode_data": "constant",
                           "order_data": 3,
                           "random_crop": True,
                           "p_el_per_sample": 0.2,
                           "p_scale_per_sample": 0.2,
                           "p_rot_per_sample": 0.2
                          }
    transform_dict = {
                      "aug1": [
                                bg.SpatialTransform(**aug1_spatial_kwargs),
                                bg.MirrorTransform(axes=(0, 1, 2)),
                                bg.GammaTransform(gamma_range=(0.7, 1.5),
                                                  invert_image=False,
                                                  per_channel=True,
                                                  retain_stats=True,
                                                  p_per_sample=0.3),
	                          ],
                     }
    # aug2
    aug2_spatial_kwargs = deepcopy(aug1_spatial_kwargs)
    aug2_spatial_kwargs["patch_size"] = (96, 160, 160)
    transform_dict["aug2"] = [bg.SpatialTransform(**aug2_spatial_kwargs),] \
                              + transform_dict["aug1"][1:]

    # aug3
    aug3_spatial_kwargs = deepcopy(aug2_spatial_kwargs)
    aug3_spatial_kwargs["random_crop"] = False
    new_transforms = [bg.SpatialTransform(**aug3_spatial_kwargs),
                      bg.BrightnessTransform(mu=101, sigma=76.9,
                                             p_per_sample=0.3),]
    # spatial, mirror, gamma, brightness
    transform_dict["aug3"] = [new_transforms[0]] + transform_dict["aug1"][1:] \
                             + [new_transforms[1]]

    # aug4
    # roicrop, spatial, mirror, gamma, brightness
    transform_dict["aug4"] = [ROICropTransform(crop_size=(96, 160, 160)),] + \
                              transform_dict["aug3"]

    # aug5
    # roicrop, spatial, mirror, gamma, brightness
    aug5_spatial_kwargs = deepcopy(aug3_spatial_kwargs)
    aug5_spatial_kwargs["patch_center_dist_from_border"] = None
    aug5_spatial_kwargs["border_cval_data"] = 0
    new_transforms = [ROICropTransform(crop_size=(96, 160, 160)),
                      bg.SpatialTransform(**aug5_spatial_kwargs),]
    # RemoveLabelTransform added to preprocessing
    transform_dict["aug5"] = new_transforms + transform_dict["aug4"][2:]
    # 2D Transforms
    # roicrop, spatial, mirror, gamma, brightness
    aug6_spatial_kwargs = deepcopy(aug3_spatial_kwargs)
    aug6_spatial_kwargs["patch_size"] = (192, 192)
    transforms_2d = [bg.SpatialTransform(**aug6_spatial_kwargs),
                     bg.MirrorTransform(axes=(0, 1)),]
    transform_dict["aug6"] = transforms_2d +  transform_dict["aug3"][2:]

    aug7_spatial_kwargs = deepcopy(aug3_spatial_kwargs)
    aug7_spatial_kwargs["patch_size"] = (256, 256)
    transforms_2d = [bg.SpatialTransform(**aug7_spatial_kwargs),
                     bg.MirrorTransform(axes=(0, 1)),]
    transform_dict["aug7"] = transforms_2d +  transform_dict["aug3"][2:]

    tu_only = [bgut.RemoveLabelTransform(1, 0),]
    transform_dict["tu_only2d"] = transform_dict["aug7"] + tu_only

    train_transform = transform_dict[augmentation_key]
    print(f"Train Transforms: {train_transform}")
    return bg.Compose(train_transform)

def get_validation_augmentation(augmentation_key):
    """
    Validation data augmentations. Usually, just cropping.
    """
    transform_dict = {
                      "aug1": [
                        bg.RandomCropTransform(crop_size=(80, 160, 160))
                      ],
                      "aug2": [
                        bg.RandomCropTransform(crop_size=(96, 160, 160))
                      ],
                      "aug3": [
                        bg.RandomCropTransform(crop_size=(96, 160, 160))
                      ],
                      "aug4": [
                        bg.RandomCropTransform(crop_size=(96, 160, 160))
                      ],
                      "aug5": [
                        ROICropTransform(crop_size=(96, 160, 160))
                      ],
                      "aug6": [
                        bg.RandomCropTransform(crop_size=(192, 192))
                      ],
                      "aug7": [
                        bg.CenterCropTransform(crop_size=(256, 256))
                      ],
                      "tu_only2d": [
                        bg.CenterCropTransform(crop_size=(256, 256)),
                        bgut.RemoveLabelTransform(1, 0)
                      ],
                     }
    test_transform = transform_dict[augmentation_key]
    print(f"\nTest/Validation Transforms: {test_transform}")
    return bg.Compose(test_transform)

def get_preprocessing(rgb: bool = False):
    """
    Construct preprocessing transform

    Args:
        rgb (bool): Whether or not to return the input with three channels
            or just single (grayscale)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        bgct.ClipValueRange(min=-79, max=304),
        bgsnt.MeanStdNormalizationTransform(mean=101, std=76.9,
                                            per_channel=False),
        bgut.RemoveLabelTransform(-1, 0),
        bg.NumpyToTensor(),
    ]
    if rgb:
        # insert right before converting to a torch tensor
        _transform.insert(-2, RepeatChannelsTransform(num_repeats=3))
    print(f"\nPreprocessing Transforms: {_transform}")
    return bg.Compose(_transform)

def parse_fg_slice_dict_single_class(json_path, out_path, removed_fg_idx="1"):
    """
    Reads the foreground (positive) class slice dictionary and creates a new
    dictionary with `removed_fg_idx` removed.
    Args:
        json_path (str): json should be 'slice_indices.json' generated by
            `io.Preprocessor` (sub dicts contain keys for list of slice indices
            for each foreground class)
        out_path (str): path to the json to save the new dictionary with
            the `removed_fg_idx` key removed
        removed_fg_idx (str): string key to remove ('1' or '2' in this case)
    Returns:
        the changed slice_dict
    """
    # reading json
    with open(json_path, "r") as fp:
        slice_dict = json.load(fp)
    cases = list(slice_dict.keys())
    # vv assumes same for all cases (which is true)
    sub_dict_keys = list(slice_dict[cases[0]])
    print(f"{len(cases)} Cases; Case sub-dict keys: {sub_dict_keys}")
    # removing all sub dicts with the `removed_fg_idx` key
    print(f"Removing idx: {removed_fg_idx}")
    for case in cases:
        case_dict = slice_dict[case]
        case_dict.pop(removed_fg_idx)
    sub_dict_keys = list(slice_dict[cases[0]])
    print(f"New case sub-dict keys: {sub_dict_keys}")
    # saving dict
    with open(out_path, "w") as fp:
        json.dump(slice_dict, fp)
    print(f"Saved at {out_path}.")
    return slice_dict

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
