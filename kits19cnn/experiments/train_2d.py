import json
from abc import abstractmethod

import torch
import segmentation_models_pytorch as smp

from kits19cnn.io import SliceDataset, PseudoSliceDataset
from kits19cnn.models import Generic_UNet
from .utils import get_training_augmentation, get_validation_augmentation, \
                   get_preprocessing
from .train import TrainExperiment, TrainClfSegExperiment

class TrainExperiment2D(TrainExperiment):
    """
    Stores the main parts of a experiment with 2D images:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    @abstractmethod
    def get_model(self):
        """
        Creates and returns the model.
        """
        return

    def get_datasets(self, train_ids, valid_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        use_rgb = "smp" in self.model_params["architecture"]
        # creating the datasets
        with open(self.io_params["slice_indices_path"], "r") as fp:
            pos_slice_dict = json.load(fp)
        p_pos_per_sample = self.io_params["p_pos_per_sample"]
        if self.io_params.get("pseudo_3D"):
            assert not use_rgb, \
                "Currently architectures that require RGB inputs cannot use pseudo slices."
            train_dataset = PseudoSliceDataset(im_ids=train_ids,
                                               pos_slice_dict=pos_slice_dict,
                                               transforms=train_aug,
                                               preprocessing=get_preprocessing(use_rgb),
                                               p_pos_per_sample=p_pos_per_sample,
                                               mode=self.config["mode"],
                                               num_pseudo_slices=self.io_params["num_pseudo_slices"])
            valid_dataset = PseudoSliceDataset(im_ids=valid_ids,
                                               pos_slice_dict=pos_slice_dict,
                                               transforms=val_aug,
                                               preprocessing=get_preprocessing(use_rgb),
                                               p_pos_per_sample=p_pos_per_sample,
                                               mode=self.config["mode"],
                                               num_pseudo_slices=self.io_params["num_pseudo_slices"])
        else:
            train_dataset = SliceDataset(im_ids=train_ids,
                                         pos_slice_dict=pos_slice_dict,
                                         transforms=train_aug,
                                         preprocessing=get_preprocessing(use_rgb),
                                         p_pos_per_sample=p_pos_per_sample,
                                         mode=self.config["mode"])
            valid_dataset = SliceDataset(im_ids=valid_ids,
                                         pos_slice_dict=pos_slice_dict,
                                         transforms=val_aug,
                                         preprocessing=get_preprocessing(use_rgb),
                                         p_pos_per_sample=p_pos_per_sample,
                                         mode=self.config["mode"])

        return (train_dataset, valid_dataset)

class TrainSegExperiment2D(TrainExperiment2D):
    """
    Stores the main parts of a segmentation experiment:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_model(self):
        architecture = self.model_params["architecture"]
        if architecture.lower() == "nnunet":
            architecture_kwargs = self.model_params[architecture]
            architecture_kwargs["norm_op"] = torch.nn.InstanceNorm2d
            architecture_kwargs["nonlin"] = torch.nn.ReLU
            architecture_kwargs["nonlin_kwargs"] = {"inplace": True}
            architecture_kwargs["final_nonlin"] = lambda x: x
            model = Generic_UNet(**architecture_kwargs)
        elif architecture.lower() == "unet_smp":
            model = smp.Unet(encoder_name=self.model_params["encoder"],
                             encoder_weights="imagenet",
                             classes=3, activation=None,
                             **self.model_params[architecture])
        elif architecture.lower() == "fpn_smp":
            model = smp.FPN(encoder_name=self.model_params["encoder"],
                            encoder_weights="imagenet",
                            classes=3, activation=None,
                            **self.model_params[architecture])
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model

class TrainClfSegExperiment2D(TrainExperiment2D, TrainClfSegExperiment):
    """
    Stores the main parts of a classification+segmentation experiment:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_model(self):
        architecture = self.model_params["architecture"]
        if architecture.lower() == "nnunet":
            architecture_kwargs = self.model_params[architecture]
            if self.io_params["batch_size"] < 10:
                architecture_kwargs["norm_op"] = torch.nn.InstanceNorm2d
            architecture_kwargs["nonlin"] = torch.nn.ReLU
            architecture_kwargs["nonlin_kwargs"] = {"inplace": True}
            architecture_kwargs["final_nonlin"] = lambda x: x
            model = Generic_UNet(**architecture_kwargs)
        else:
            raise NotImplementedError
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model
