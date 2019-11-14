import json
import torch
import segmentation_models_pytorch as smp

from kits19cnn.io import SliceDataset
from kits19cnn.models import Generic_UNet
from .utils import get_training_augmentation, get_validation_augmentation, \
                   get_preprocessing
from .train import TrainExperiment

class TrainSeg2dExperimentFromConfig(TrainExperiment):
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

        train_dataset = SliceDataset(im_ids=train_ids,
                                     pos_slice_dict=pos_slice_dict,
                                     transforms=train_aug,
                                     preprocessing=get_preprocessing(use_rgb),
                                     p_pos_per_sample=p_pos_per_sample)
        valid_dataset = SliceDataset(im_ids=valid_ids,
                                     pos_slice_dict=pos_slice_dict,
                                     transforms=val_aug,
                                     preprocessing=get_preprocessing(use_rgb),
                                     p_pos_per_sample=p_pos_per_sample)

        return (train_dataset, valid_dataset)

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
