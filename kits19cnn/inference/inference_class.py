from pathlib import Path
from os.path import join, isdir
from tqdm import tqdm
import os
import numpy as np
import inspect
import nibabel as nib
import torch

from kits19cnn.inference.utils import load_weights_infer

class Predictor(object):
    """
    Inference for a single model for every file generated by `test_loader`.
    Predictions are saved in `out_dir`.
    """
    def __init__(self, out_dir, checkpoint_path, model,
                 test_loader, pred_3D_params={"do_mirroring": True},
                 pseudo_3D: bool = False):
        """
        Attributes
            out_dir (str): path to the output directory to store predictions
            checkpoint_path (str): path to a model checkpoint for `model`
            model (torch.nn.Module): class with the `predict_3D` method for
                predicting a single patient volume.
            test_loader: Iterable instance for generating data
                (pref. torch DataLoader)
                must have the __len__ arg.
            pred_3D_params (dict): kwargs for `model.predict_3D`
            pseudo_3D (bool): whether or not to have pseudo 3D inputs
        """
        self.out_dir = out_dir
        if not isdir(self.out_dir):
            os.mkdir(self.out_dir)
            print(f"Created {self.out_dir}!")
        assert inspect.ismethod(model.predict_3D), \
                "model must have the method `predict_3D`"
        self.model = load_weights_infer(checkpoint_path, model)
        self.test_loader = test_loader
        self.pred_3D_params = pred_3D_params
        self.pseudo_3D = pseudo_3D

    def run_3D_predictions(self):
        """
        Runs predictions on the dataset (specified in test_loader)
        """
        cases = self.test_loader.dataset.im_ids
        assert len(cases) == len(self.test_loader)
        for (test_batch, case) in tqdm(zip(self.test_loader, cases), total=len(cases)):
            test_x = torch.squeeze(test_batch[0], dim=0)
            if self.pseudo_3D:
                pred, _, act, _ = self.model.predict_3D_pseudo3D_2Dconv(test_x,
                                                                    **self.pred_3D_params)
            else:
                pred, _, act, _ = self.model.predict_3D(test_x,
                                                        **self.pred_3D_params)
            assert len(pred.shape) == 3
            assert len(act.shape) == 4
            ### possible place to threshold ROI size ###
            self.save_pred(pred, act, case)

    def save_pred(self, pred, act, case):
        """
        Saves both prediction and activation maps in `out_dir` in the
        KiTS19 format.
        Args:
            pred (np.ndarray): shape (x, y, z)
            act (np.ndarray): shape (n_classes, x, y, z)
            case: path to a case folder (an element of self.cases)
        Returns:
            None
        """
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        np.save(join(out_case_dir, "pred.npy"), pred)
        np.save(join(out_case_dir, "pred_act.npy"), act)

    def resample_predictions(self, orig_spacing, target_spacing,
                             resampled_preds_dir):
        """
        Iterates through `out_dir` and creates resampled .npy arrays to
        the specified spacing and saves them in `resampled_preds_dir`
        """
        from kits19cnn.io.resample import resample_patient
        raise NotImplementedError

    def prepare_submission(self):
        """
        Resamples predictions and converts them to a .zip with .nii.gz files.
        """
        raise NotImplementedError
