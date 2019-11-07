from glob import glob
from tqdm import tqdm
from abc import abstractmethod
import os

import torch
from torch.data.utils import DataLoader
from sklearn.model_selection import train_test_split

from kits19cnn.io import TestVoxelDataset
from kits19cnn.utils import softmax_helper
from utils import get_preprocessing

class BaseInferenceExperiment(object):
    def __init__(self, config: dict):
        """
        Args:
            config (dict):

        Attributes:
            config-related:
                config (dict):
                io_params (dict):
                    in_dir (key: str): path to the data folder
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
            split_dict (dict): test_ids
            test_dset (torch.data.Dataset): <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
        """
        # for reuse
        self.config = config
        self.io_params = config["io_params"]
        # initializing the experiment components
        self.case_list = self.setup_im_ids()
        test_ids = self.get_split()[-1] if config["with_masks"] else self.case_list
        print(f"Inferring on {len(test_ids)} test cases")
        self.test_dset = self.get_datasets(test_ids)
        self.loaders = self.get_loaders()
        self.model = self.get_model()

    @abstractmethod
    def get_datasets(self, test_ids):
        """
        Initializes the data augmentation and preprocessing transforms. Creates
        and returns the train and validation datasets.
        """
        return

    @abstractmethod
    def get_model(self):
        """
        Creates and returns the model.
        """
        return

    def setup_im_ids(self):
        """
        Creates a list of all paths to case folders for the dataset split
        """
        search_path = os.path.join(self.config["in_dir"], "*/")
        case_list = sorted(glob(search_path))
        case_list = case_list[:210] if self.config["with_masks"] else case_list[210:]
        return case_list

    def get_split(self):
        """
        Creates train/valid filename splits
        """
        # setting up the train/val split with filenames
        split_seed: int = self.io_params["split_seed"]
        test_size: float = self.io_params["test_size"]
        # doing the splits: 1-test_size, test_size//2, test_size//2
        print("Splitting the dataset normally...")
        train_ids, total_test = train_test_split(self.case_list,
                                                 random_state=split_seed,
                                                 test_size=test_size)
        val_ids, test_ids = train_test_split(sorted(total_test),
                                             random_state=split_seed,
                                             test_size=0.5)
        return (train_ids, val_ids, test_ids)

    def get_loaders(self):
        """
        Creates train/val loaders from datasets created in self.get_datasets.
        Returns the loaders.
        """
        # setting up the loaders
        b_size, num_workers = self.io_params["batch_size"], self.io_params["num_workers"]
        test_loader = DataLoader(self.test_dset, batch_size=b_size,
                                  shuffle=True, num_workers=num_workers)
        return {"test": test_loader}

class SegmentationInferenceExperiment(BaseInferenceExperiment):
    """
    Inference Experiment to support prediction experiments
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict):
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_datasets(self, test_ids):
        """
        Creates and returns the test dataset.
        """
        # creating the datasets
        test_dataset = TestVoxelDataset(im_ids=test_ids,
                                        transforms=None,
                                        preprocessing=get_preprocessing(),
                                        file_ending=self.io_params["file_ending"])
        return test_dataset

    def get_model(self):
        architecture = self.model_params["architecture"]
        # creating model
        if architecture == "nnunet":
            unet_kwargs = self.model_params[architecture]
            unet_kwargs = self.setup_3D_UNet_params(unet_kwargs)
            model = Generic_UNet(**unet_kwargs)
            model.inference_apply_nonlin = softmax_helper
        else:
            raise NotImplementedError
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model

    def setup_3D_UNet_params(self, unet_kwargs):
        """
        ^^^^^^^^^^^^^
        """
        unet_kwargs["conv_op"] = torch.nn.Conv3d
        unet_kwargs["norm_op"] = torch.nn.InstanceNorm3d
        unet_kwargs["dropout_op"] = torch.nn.Dropout3d
        unet_kwargs["nonlin"] = torch.nn.ReLU
        unet_kwargs["nonlin_kwargs"] = {"inplace": True}
        unet_kwargs["final_nonlin"] = lambda x: x
        return unet_kwargs
