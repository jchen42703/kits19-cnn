from tqdm import tqdm
import numpy as np
import inspect
import nibabel as nib
import torch

from kits19cnn.inference.utils import load_weights_infer

class Predictor(object):
    """
    Inference for a single model for every file in `cases` in `in_dir`.
    Predictions are saved in `out_dir`.
    """
    def __init__(self, in_dir, out_dir, cases, checkpoint_path, model,
                 test_loader, pred_3D_params={"do_mirroring": True}):
        assert inspect.ismethod(model.predict_3D), \
                "model must have the method `predict_3D`"
        self.model = load_weights_infer(checkpoint_path, model)
        self.test_loader = test_loader
        self.pred_3D_params = pred_3D_params

    def run_3D_predictions(self):
        """
        Runs predictions on the dataset (specified in test_loader)
        """
        for test_batch in tqdm(self.test_loader):
            test_x = torch.squeeze(test_batch[0], dim=0)
            pred, act = self.model.predict_3D(test_x, **self.pred_3D_params)
            assert len(pred.shape) == 3
            assert len(act.shape) == 4 # temp checks
            ### possible place to threshold ROI size ###
            self.save_pred(pred, act)

    def save_pred(pred, act):
        """
        Saves both prediction and activation maps in `out_dir` in the
        KiTS19 format.
        Args:
            pred (np.ndarray): shape (x, y, z)
            act (np.ndarray): shape (n_classes, x, y, z)
        Returns:
            None
        """
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        np.save(os.path.join(out_case_dir, "pred.npy"), image)
        np.save(os.path.join(out_case_dir, "pred_act.npy"), mask)

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
