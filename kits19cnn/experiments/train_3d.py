import torch

from kits19cnn.models import Generic_UNet
from kits19cnn.io import VoxelDataset, ClfSegVoxelDataset

from .train import TrainExperiment, TrainClfSegExperiment
from .utils import get_preprocessing, get_training_augmentation, \
                  get_validation_augmentation, seed_everything

class TrainSegExperiment(TrainExperiment):
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
        # creating the datasets
        train_dataset = VoxelDataset(im_ids=train_ids,
                                     transforms=train_aug,
                                     preprocessing=get_preprocessing())
        valid_dataset = VoxelDataset(im_ids=valid_ids,
                                     transforms=val_aug,
                                     preprocessing=get_preprocessing())
        return (train_dataset, valid_dataset)

    def get_model(self):
        architecture = self.model_params["architecture"]
        if architecture == "nnunet":
            architecture_kwargs = self.model_params[architecture]
            architecture_kwargs["conv_op"] = torch.nn.Conv3d
            architecture_kwargs["norm_op"] = torch.nn.InstanceNorm3d
            architecture_kwargs["dropout_op"] = torch.nn.Dropout3d
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

class TrainClfSegExperiment3D(TrainClfSegExperiment, TrainSegExperiment):
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

    def get_datasets(self, train_ids, valid_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        # creating the datasets
        preprocess = get_preprocessing()
        train_dataset = ClfSegVoxelDataset(im_ids=train_ids,
                                           transforms=train_aug,
                                           preprocessing=preprocess,
                                           mode="both", num_classes=3)
        valid_dataset = ClfSegVoxelDataset(im_ids=valid_ids,
                                           transforms=val_aug,
                                           preprocessing=preprocess,
                                           mode="both", num_classes=3)
        return (train_dataset, valid_dataset)
