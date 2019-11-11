import os
import torch

from glob import glob
from abc import abstractmethod
from pathlib import Path
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback, \
                                  CheckpointCallback \
                                  # PrecisionRecallF1ScoreCallback
from catalyst.dl.runner import SupervisedRunner

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, \
                                     CosineAnnealingWarmRestarts, CyclicLR

from kits19cnn.models import Generic_UNet
from kits19cnn.io import ClfSegVoxelDataset, VoxelDataset
from kits19cnn.loss_functions import DC_and_CE_loss, BCEDiceLoss
from utils import get_preprocessing, get_training_augmentation, \
                  get_validation_augmentation, seed_everything

class TrainExperiment(object):
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_classification_yaml.py`

        Attributes:
            config-related:
                config (dict): from `train_classification_yaml.py`
                io_params (dict): contains io-related parameters
                    image_folder (key: str): path to the image folder
                    df_setup_type (key: str): regular or pos_only
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
                    aug_key (key: str): One of the augmentation keys for
                        `get_training_augmentation` and `get_validation_augmentation`
                        in `scripts/utils.py`
                opt_params (dict): optimizer related parameters
                    lr (key: str): learning rate
                    opt (key: str): optimizer name
                        Currently, only supports sgd and adam.
                    scheduler_params (key: str): dict of:
                        scheduler (key: str): scheduler name
                        {scheduler} (key: dict): args for the above scheduler
                cb_params (dict):
                    earlystop (key: str):
                        dict -> kwargs for EarlyStoppingCallback
                    accuracy (key: str):
                        dict -> kwargs for AccuracyCallback
                    checkpoint_params (key: dict):
                      checkpoint_path (key: str): path to the checkpoint
                      checkpoint_mode (key: str): model_only or
                        full (for stateful loading)
            split_dict (dict): train_ids and valid_ids
            train_dset, val_dset: <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
            opt (torch.optim.Optimizer): <-
            lr_scheduler (torch.optim.lr_scheduler): <-
            criterion (torch.nn.Module): <-
            cb_list (list): list of catalyst callbacks
        """
        # for reuse
        self.config = config
        self.io_params = config["io_params"]
        self.opt_params = config["opt_params"]
        self.cb_params = config["callback_params"]
        self.criterion_params = config["criterion_params"]
        # initializing the experiment components
        self.case_list = self.setup_im_ids()
        train_ids, val_ids, _ = self.get_split()
        self.train_dset, self.val_dset = self.get_datasets(train_ids, val_ids)
        self.loaders = self.get_loaders()
        self.model = self.get_model()
        self.opt = self.get_opt()
        self.lr_scheduler = self.get_lr_scheduler()
        self.criterion = self.get_criterion()
        self.cb_list = self.get_callbacks()

    @abstractmethod
    def get_datasets(self, train_ids, valid_ids):
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
        search_path = os.path.join(self.config["data_folder"], "*/")
        case_list = sorted(glob(search_path))
        case_list = case_list[:210] if len(case_list) >= 210 else case_list
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
        train_loader = DataLoader(self.train_dset, batch_size=b_size,
                                  shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(self.val_dset, batch_size=b_size,
                                  shuffle=False, num_workers=num_workers)

        self.train_steps = len(self.train_dset) # for schedulers
        return {"train": train_loader, "valid": valid_loader}

    def get_opt(self):
        assert isinstance(self.model, torch.nn.Module), \
            "`model` must be an instance of torch.nn.Module`"
        # fetching optimizers
        lr = self.opt_params["lr"]
        opt_name = self.opt_params["opt"].lower()
        if opt_name == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                  self.model.parameters()),
                                  lr=lr, momentum=0.9, weight_decay=0.0001)
        return opt

    def get_lr_scheduler(self):
        assert isinstance(self.opt, torch.optim.Optimizer), \
            "`optimizer` must be an instance of torch.optim.Optimizer"
        sched_params = self.opt_params["scheduler_params"]
        scheduler_name = sched_params["scheduler"].lower()
        scheduler_args = sched_params[scheduler_name]
        # fetching lr schedulers
        if scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(self.opt, **scheduler_args)
        elif scheduler_name == "cosineannealing":
            scheduler = CosineAnnealingLR(self.opt, **scheduler_args)
        elif scheduler_name == "cosineannealingwr":
            scheduler = CosineAnnealingWarmRestarts(self.opt,
                                                    **scheduler_args)
        elif scheduler_name == "clr":
            scheduler = CyclicLR(self.opt, **scheduler_args)
        print(f"LR Scheduler: {scheduler}")

        return scheduler

    def get_criterion(self):
        loss_name = self.criterion_params["loss"].lower()
        if loss_name == "bce_dice_loss":
            raise NotImplementedError
            # criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        elif loss_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif loss_name == "ce_dice_loss":
            # for softmax
            soft_dice_kwargs = {"batch_dice": True, "smooth": 1e-5,
                                "do_bg": False, "square": False}
            criterion = DC_and_CE_loss(soft_dice_kwargs=soft_dice_kwargs,
                                       ce_kwargs={})
        print(f"Criterion: {criterion}")

        return criterion

    def get_callbacks(self):
        callbacks_list = [#PrecisionRecallF1ScoreCallback(num_classes=3),#DiceCallback(),
                          # DiceCallback()
                          EarlyStoppingCallback(**self.cb_params["earlystop"]),
                          # AccuracyCallback(**self.cb_params["accuracy"]),
                          ]
        ckpoint_params = self.cb_params["checkpoint_params"]
        if ckpoint_params["checkpoint_path"] != None: # hacky way to say no checkpoint callback but eh what the heck
            mode = ckpoint_params["mode"].lower()
            if mode == "full":
                print("Stateful loading...")
                ckpoint_p = Path(ckpoint_params["checkpoint_path"])
                fname = ckpoint_p.name
                # everything in the path besides the base file name
                resume_dir = str(ckpoint_p.parents[0])
                print(f"Loading {fname} from {resume_dir}. \
                      \nCheckpoints will also be saved in {resume_dir}.")
                # adding the checkpoint callback
                callbacks_list = callbacks_list + [CheckpointCallback(resume=fname,
                                                                      resume_dir=resume_dir),]
            elif mode == "model_only":
                print("Loading weights into model...")
                self.model = load_weights_train(ckpoint_params["checkpoint_path"], self.model)
        return callbacks_list

class TrainSegExperimentFromConfig(TrainExperiment):
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

class TrainClfSegExperimentFromConfig(TrainSegExperimentFromConfig):
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

    def get_criterion(self):
        loss_dict = {
            "bce_dice_loss": BCEDiceLoss(eps=1.),
            "bce": torch.nn.BCEWithLogitsLoss(),
            "ce_dice_loss": DC_and_CE_loss(soft_dice_kwargs={}, ce_kwargs={}),
        }

        seg_loss_name = self.criterion_params["seg_loss"].lower()
        clf_loss_name = self.criterion_params["clf_loss"].lower()

        criterion_dict = {seg_loss_name: loss_dict[seg_loss_name],
                          clf_loss_name: loss_dict[clf_loss_name]}
        print(f"Criterion: {criterion_dict}")
        return criterion_dict

    def get_callbacks(self):
        from catalyst.dl.callbacks import CriterionAggregatorCallback, \
                                          CriterionCallback

        callbacks_list = [
                          CriterionCallback(prefix="seg_loss",
                                            input_key="seg_targets",
                                            output_key="seg_logits",
                                            criterion_key="ce_dice_loss"),
                          CriterionCallback(prefix="clf_loss",
                                            input_key="clf_targets",
                                            output_key="clf_logits",
                                            criterion_key="bce_dice_loss"),
                          CriterionAggregatorCallback(prefix="loss",
                                                      loss_keys=\
                                                      ["seg_loss", "clf_loss"]),
                          EarlyStoppingCallback(**self.cb_params["earlystop"]),
                          ]

        ckpoint_params = self.cb_params["checkpoint_params"]
        if ckpoint_params["checkpoint_path"] != None: # hacky way to say no checkpoint callback but eh what the heck
            mode = ckpoint_params["mode"].lower()
            if mode == "full":
                print("Stateful loading...")
                ckpoint_p = Path(ckpoint_params["checkpoint_path"])
                fname = ckpoint_p.name
                # everything in the path besides the base file name
                resume_dir = str(ckpoint_p.parents[0])
                print(f"Loading {fname} from {resume_dir}. \
                      \nCheckpoints will also be saved in {resume_dir}.")
                # adding the checkpoint callback
                callbacks_list = callbacks_list + [CheckpointCallback(resume=fname,
                                                                      resume_dir=resume_dir),]
            elif mode == "model_only":
                print("Loading weights into model...")
                self.model = load_weights_train(ckpoint_params["checkpoint_path"], self.model)
        print(f"Callbacks: {callbacks_list}")
        return callbacks_list

def load_weights_train(checkpoint_path, model):
    """
    Loads weights from a checkpoint and into training.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in train() mode
    """
    try:
        # catalyst weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    except:
        # anything else
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.train()
    return model
