import os
import random
import numpy as np
import batchgenerators.transforms as bg
import torch

from kits19cnn.io import ROICropTransform

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
    transform_dict["aug3"] = [new_transforms[0]] + transform_dict["aug1"][1:] \
                             + [new_transforms[1]]

    # aug4
    transform_dict["aug4"] = [ROICropTransform(crop_size=(96, 160, 160)),] + \
                              transform_dict["aug3"]

    train_transform = transform_dict[augmentation_key]
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
                     }
    test_transform = transform_dict[augmentation_key]
    return bg.Compose(test_transform)

def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    bgct = bg.color_transforms
    bgsnt = bg.sample_normalization_transforms
    _transform = [
        bgct.ClipValueRange(min=-79, max=304),
        bgsnt.MeanStdNormalizationTransform(mean=101, std=76.9,
                                            per_channel=False),
        bg.NumpyToTensor(),
    ]
    return bg.Compose(_transform)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
