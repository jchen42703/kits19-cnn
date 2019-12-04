import os
from glob import glob
from abc import abstractmethod
from pathlib import Path
import catalyst.dl.callbacks as callbacks
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from kits19cnn.loss_functions import DC_and_CE_loss, BCEDiceLoss
from .utils import get_preprocessing, get_training_augmentation, \
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
        """
        Creates the optimizer
        """
        assert isinstance(self.model, torch.nn.Module), \
            "`model` must be an instance of torch.nn.Module`"
        # fetching optimizers
        opt_name = self.opt_params["opt"]
        opt_kwargs = self.opt_params[opt_name]
        opt_cls = torch.optim.__dict__[opt_name]
        opt = opt_cls(filter(lambda p: p.requires_grad,
                             self.model.parameters()),
                      **opt_kwargs)
        print(f"Optimizer: {opt}")
        return opt

    def get_lr_scheduler(self):
        """
        Creates the LR scheduler from the optimizer created in `self.get_opt`
        """
        assert isinstance(self.opt, torch.optim.Optimizer), \
            "`optimizer` must be an instance of torch.optim.Optimizer"
        sched_params = self.opt_params["scheduler_params"]
        scheduler_name = sched_params["scheduler"]
        scheduler_args = sched_params[scheduler_name]
        scheduler_cls = torch.optim.lr_scheduler.__dict__[scheduler_name]
        scheduler = scheduler_cls(optimizer=self.opt, **scheduler_args)
        print(f"LR Scheduler: {scheduler.__class__.__name__}")
        return scheduler

    def get_criterion(self):
        """
        Fetches the criterion. (Only one loss.)
        """
        loss_name = self.criterion_params["loss"].lower()
        loss_dict = {
            "bce_dice_loss": BCEDiceLoss(eps=1.),
            "bce": torch.nn.BCEWithLogitsLoss(),
            "ce_dice_loss": DC_and_CE_loss(soft_dice_kwargs={}, ce_kwargs={}),
        }
        # re-initializing criterion with kwargs
        loss_kwargs = self.criterion_params.get(loss_name)
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs

        loss = loss_dict[loss_name]
        loss.__init__(**loss_kwargs)
        print(f"Criterion: {loss}")
        return loss

    def get_callbacks(self):
        """
        Creates a list of callbacks.
        """
        cb_name_list = list(self.cb_params.keys())
        cb_name_list.remove("checkpoint_params")
        callbacks_list = [callbacks.__dict__[cb_name](**self.cb_params[cb_name])
                          for cb_name in cb_name_list]
        callbacks_list = self.load_weights(callbacks_list)
        print(f"Callbacks: {[cb.__class__.__name__ for cb in callbacks_list]}")
        return callbacks_list

    def load_weights(self, callbacks_list):
        """
        Loads model weights and appends the CheckpointCallback if doing
        stateful model loading. This doesn't add the CheckpointCallback if
        it's 'model_only' loading bc SupervisedRunner adds it by default.
        """
        ckpoint_params = self.cb_params["checkpoint_params"]
        # Having checkpoint_params=None is a hacky way to say no checkpoint
        # callback but eh what the heck
        if ckpoint_params["checkpoint_path"] != None:
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
                ckpoint = [callbacks.CheckpointCallback(resume=fname,
                                                        resume_dir=resume_dir)]
                callbacks_list = callbacks_list + ckpoint
            elif mode == "model_only":
                print("Loading weights into model...")
                self.model = load_weights_train(ckpoint_params["checkpoint_path"],
                                                self.model)
        return callbacks_list

class TrainClfSegExperiment(TrainExperiment):
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
            config (dict): from a .yml file.
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

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

    def get_criterion(self):
        """
        Returns a dictionary of the desired criterion (for seg and clf)
        """
        loss_dict = {
            "bce_dice_loss": BCEDiceLoss(eps=1.),
            "bce": torch.nn.BCEWithLogitsLoss(),
            "ce_dice_loss": DC_and_CE_loss(soft_dice_kwargs={}, ce_kwargs={}),
        }

        seg_loss_name = self.criterion_params["seg_loss"].lower()
        clf_loss_name = self.criterion_params["clf_loss"].lower()

        # re-initializing criterion with kwargs
        seg_kwargs = self.criterion_params.get(seg_loss_name)
        clf_kwargs = self.criterion_params.get(clf_loss_name)
        seg_kwargs = {} if seg_kwargs is None else seg_kwargs
        clf_kwargs = {} if clf_kwargs is None else clf_kwargs

        seg_loss = loss_dict[seg_loss_name]
        clf_loss = loss_dict[clf_loss_name]
        seg_loss.__init__(**seg_kwargs), clf_loss.__init__(**clf_kwargs)
        criterion_dict = {seg_loss_name: seg_loss,
                          clf_loss_name: clf_loss}
        print(f"Criterion: {criterion_dict}")
        return criterion_dict

    def get_callbacks(self):
        """
        Gets the callbacks list; since this is multi-task, we need multiple
        metrics! Therefore, callbacks_list will now contain the
        CriterionAggregatorCallback and CriterionCallback. They calculate and
        record the `seg_loss` and `clf_loss`.
        """
        from catalyst.dl.callbacks import CriterionAggregatorCallback, \
                                          CriterionCallback
        seg_loss_name = self.criterion_params["seg_loss"].lower()
        clf_loss_name = self.criterion_params["clf_loss"].lower()
        criterion_cb_list = [
                          CriterionCallback(prefix="seg_loss",
                                            input_key="seg_targets",
                                            output_key="seg_logits",
                                            criterion_key=seg_loss_name),
                          CriterionCallback(prefix="clf_loss",
                                            input_key="clf_targets",
                                            output_key="clf_logits",
                                            criterion_key=clf_loss_name),
                          CriterionAggregatorCallback(prefix="loss",
                                                      loss_keys=\
                                                      ["seg_loss", "clf_loss"]),
                          ]
        # regular callbacks
        cb_name_list = list(self.cb_params.keys())
        cb_name_list.remove("checkpoint_params")
        callbacks_list = [callbacks.__dict__[cb_name](**self.cb_params[cb_name])
                          for cb_name in cb_name_list]
        callbacks_list = self.load_weights(callbacks_list) + criterion_cb_list
        print(f"Callbacks: {[cb.__class__.__name__ for cb in callbacks_list]}")
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
