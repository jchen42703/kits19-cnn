from tqdm import tqdm
import numpy as np
import nibabel as nib
import pandas as pd
import os
from os.path import isdir, join
from pathlib import Path

from kits19cnn.metrics import evaluate_official
from sklearn.metrics import precision_recall_fscore_support

class Evaluator(object):
    """
    Evaluates all of the predictions in a user-specified directory and logs them in a csv. Assumes that
    the output is in the KiTS19 file structure.
    """
    def __init__(self, orig_img_dir, pred_dir, cases=None,
                 label_file_ending=".npy", binary_tumor=False):
        """
        Attributes:
            orig_img_dir: path to the directory containing the
                labels to evaluate with
                i.e. original kits19/data directory or the preprocessed imgs
                directory
                assumes structure:
                orig_img_dir
                    case_xxxxx
                        imaging{file_ending}
                        segmentation{file_ending}
            pred_dir: path to the predictions directory, created by Predictor
                assumes structure:
                pred_dir
                    case_xxxxx
                        pred.npy
                        act.npy
            cases: list of filepaths to case folders or just case folder names.
                Defaults to None.
            label_file_ending (str): one of ['.npy', '.nii', '.nii.gz']
            binary_tumor (bool): whether or not to treat predicted 1s as tumor
        """
        self.orig_img_dir = orig_img_dir
        self.pred_dir = pred_dir
        self.file_ending = label_file_ending
        assert self.file_ending in [".npy", ".nii", ".nii.gz"], \
            "label_file_ending must be one of [''.npy', '.nii', '.nii.gz']"
        # converting cases from filepaths to raw folder names
        if cases is None:
            self.cases_raw = [case \
                              for case in os.listdir(self.pred_dir) \
                              if case.startswith("case")]
            assert len(self.cases_raw) > 0, \
                "Please make sure that pred_dir has the case folders"
        elif cases is not None:
            # extracting raw cases from filepath cases
            cases_raw = [Path(case).name for case in cases]
            # filtering them down to only cases in pred_dir
            self.cases_raw = [case for case in cases_raw \
                              if isdir(join(self.pred_dir, case))]
        self.binary_tumor = binary_tumor
        if self.binary_tumor:
            print("Evaluating predicted 1s as tumor (changed to 2).")

    def evaluate_all(self, print_metrics=False):
        """
        Evaluates all cases and creates the results.csv, which stores all of
        the metrics and the averages.
        Args:
            print_metrics (bool): whether or not to print metrics.
                Defaults to False to be cleaner with tqdm.
        """
        metrics_dict = {"cases": [],
                        "tk_dice": [], "tu_dice": [],
                        "precision": [], "recall": [],
                        "fpr": [], "orig_shape": [],
                        "support": [], "pred_support": []}

        for case in tqdm(self.cases_raw):
            # loading the necessary arrays
            label, pred = self.load_masks_and_pred(case)
            metrics_dict = self.eval_all_metrics_per_case(metrics_dict, label,
                                                          pred, case,
                                                          print_metrics)

        metrics_dict = self.round_all(self.average_all_cases_per_metric(metrics_dict))
        df = pd.DataFrame(metrics_dict)
        metrics_path = join(self.pred_dir, "results.csv")
        print(f"Saving {metrics_path}...")
        df.to_csv(metrics_path)

    def load_masks_and_pred(self, case):
        """
        Loads mask and prediction from `case`
        Args:
            case (str): case folder names to use
        Returns:
            label (np.ndarray): shape (x, y, z)
            pred (np.ndarray): shape (x, y, z)
        """
        y_path = join(self.orig_img_dir, case, f"segmentation{self.file_ending}")
        if self.file_ending == ".npy":
            label = np.load(y_path)
        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            label = nib.load(y_path).get_fdata()
        pred = np.load(join(self.pred_dir, case, "pred.npy")).squeeze()
        if self.binary_tumor:
            # treating prediced 1s as tumor (2)
            pred[pred == 1] = 2
        return (label, pred)

    def eval_all_metrics_per_case(self, metrics_dict, y_true, y_pred,
                                  case, print_metrics=False):
        """
        Calculates the official metrics, precision, recall, specificity (fpr),
        and stores some metadata such as the original shape and support
        (# of pixels for each class). They are then appended to the main
        metrics dictionary that is to become results.csv.
        """
        # calculating metrics
        tk_dice, tu_dice = evaluate_official(y_true, y_pred)
        prec, recall, _, supp = precision_recall_fscore_support(y_true.ravel(),
                                                                y_pred.ravel(),
                                                                labels=[0, 1, 2])
        pred_supp = np.unique(y_pred, return_counts=True)[-1]
        fpr = 1-recall
        orig_shape = y_true.shape

        if print_metrics:
            print(f"PPV: {prec}\nTPR: {recall}\nSupp: {supp}")
            print(f"Tumour and Kidney Dice: {tk_dice}; Tumour Dice: {tu_dice}")
        # order for appending (sorted keys)
        # ['cases', 'fpr', 'orig_shape', 'precision', 'pred_support', 'recall',
        # 'support', 'tk_dice', 'tu_dice']
        append_list = [case, fpr, orig_shape, prec, pred_supp, recall, supp,
                       tk_dice, tu_dice]
        sorted_keys = sorted(metrics_dict.keys())
        assert len(append_list) == len(sorted_keys)
        # appending to each key's list
        for (key_, value_) in zip(sorted_keys, append_list):
            metrics_dict[key_].append(value_)
        return metrics_dict

    def average_all_cases_per_metric(self, metrics_dict):
        """
        Averages the metrics (each key of metrics_dict).
        """
        metrics_dict["cases"].append("average")
        for key in list(metrics_dict.keys()):
            if key == "cases":
                pass
            else:
                # axis=0 will make it so that each sub-axis of orig_shape and
                # support will be averaged
                try:
                    metrics_dict[key].append(np.mean(metrics_dict[key], axis=0))
                except:
                    metrics_dict[key].append("N/A")
        return metrics_dict

    def round_all(self, metrics_dict):
        """
        Rounding all relevant metrics to three decimal places for cleanliness.
        """
        for key in list(metrics_dict.keys()):
            if key in ["cases", "pred_support"]:
                pass
            else:
                metrics_dict[key] = np.round(metrics_dict[key],
                                             decimals=3).tolist()
        return metrics_dict
