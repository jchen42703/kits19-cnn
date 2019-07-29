import numpy as np
import nibabel as nib
import pandas as pd

from os.path import isdir, join
from kits19cnn.models.metrics import evaluate_official
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

class Evaluator(object):
    """
    Evaluates all of the predictions in a user-specified directory and logs them in a csv. Assumes that
    the output is in the KiTS19 file structure.
    """
    def __init__(self, orig_img_dir, pred_dir, cases=None, binary_label=None):
        """
        Attributes:
            orig_img_dir: path to the original kits19/data directory
            pred_dir: path to the predictions directory, created by Predictor
            cases: list of filepaths to case folders or just case folder names
            binary_label: label (not including background class) to evaluate. Defaults to None.
                * If None, Evaluator does multi-class evaluation.
                F-score results are logged in the tu_dice column. tk_dice is automatically set to 0.
        """
        self.orig_img_dir = orig_img_dir
        self.pred_dir = pred_dir
        # converting cases from filepaths to raw folder names
        if cases is None:
            self.cases_raw = [case \
                              for case in os.listdir(self.pred_dir) \
                              if case.startswith("case")]
            assert len(self.cases) > 0, "Please make sure that orig_img_dir refers to the kits19/data directory."
        elif cases is not None:
            # extracting raw cases from filepath cases
            cases_raw = [Path(case).name for case in cases]
            # filtering them down to only cases in pred_dir
            self.cases_raw = [case for case in cases_raw if isdir(join(self.pred_dir, case))]
        self.binary_label = binary_label

    def evaluate_all(self):
        """
        Evaluates all cases and creates the results.csv, which stores all of the metrics and the averages.
        """
        metrics_dict = {"cases": [],
                        "tk_dice": [], "tu_dice": [],
                        "precision": [], "recall": [],
                        "fpr": [], "orig_shape": [],
                        "support": [], "pred_support": []}

        for (i, case) in enumerate(self.cases_raw):
            print("Evaluating {0}/{1}: {2}".format(i+1, len(self.cases_raw), case))
            # loading the necessary arrays
            label = nib.load(join(self.orig_img_dir, case, "segmentation.nii.gz")).get_fdata()
            if self.binary_label is not None:
                # adjusting labels for binary evaluation
                label[label != self.binary_label] = 0
                label[label == self.binary_label] = 1
            save_name = "pred_{0}.npy".format(case)
            pred = np.load(join(self.pred_dir, case, save_name))
            metrics_dict = self.evaluate_with_all_metrics_per_case(metrics_dict, label, pred, case)

        metrics_dict = self.round_all(self.average_all_cases_per_metric(metrics_dict))
        df = pd.DataFrame(metrics_dict)
        df.to_csv(join(self.pred_dir, "results.csv"))
        print("Done!")

    def average_all_cases_per_metric(self, metrics_dict):
        """
        Averages the metrics (each key of metrics_dict).
        """
        metrics_dict["cases"].append("average")
        for key in list(metrics_dict.keys()):
            if key == "cases":
                pass
            else:
                # axis=0 will make it so that each sub-axis of orig_shape and support will be averaged
                metrics_dict[key].append(np.mean(metrics_dict[key], axis=0))
        return metrics_dict

    def evaluate_with_all_metrics_per_case(self, metrics_dict, y_true, y_pred, case):
        """
        Calculates the official metrics, precision, recall, specificity (fpr), and stores some
        metadata such as the original shape and support (# of pixels for each class). They are then appended
        to the main metrics dictionary that is to become results.csv.
        """
        # calculating metrics
        labels = [0, 1, 2] if self.binary_label is None else [0, 1]
        prec, recall, fscore, supp = precision_recall_fscore_support(y_true.ravel(), y_pred.ravel(), labels=labels)
        if self.binary_label is None:
            tk_dice, tu_dice = evaluate_official(y_true, y_pred)
        else:
            # Not pretty, but logging binary dice results in the tu_dice column
            tk_dice, tu_dice = (0, fscore)
        
        pred_supp = np.unique(y_pred, return_counts=True)[-1]
        print("precision: {1}\nrecall: {2}\nsupport: {3}".format(case, prec, recall, supp))
        print("Tumour and Kidney Dice: {0}; Tumour Dice: {1}".format(tk_dice, tu_dice))
        print("Shape: {0}\n".format(y_true.shape))
        assert y_true.shape == y_pred.shape

        # appending to each key's list
        metrics_dict["cases"].append(case)
        metrics_dict["tk_dice"].append(tk_dice), metrics_dict["tu_dice"].append(tu_dice)
        metrics_dict["precision"].append(prec), metrics_dict["recall"].append(recall)
        metrics_dict["fpr"].append(1-recall), metrics_dict["orig_shape"].append(y_true.shape),
        metrics_dict["support"].append(supp), metrics_dict["pred_support"].append(pred_supp)

        return metrics_dict

    def round_all(self, metrics_dict):
        """
        Rounding all relevant metrics to three decimal places for cleanliness.
        """
        for key in list(metrics_dict.keys()):
            if key in ["cases", "orig_shape", "support", "pred_support"]:
                pass
            else:
                metrics_dict[key] = np.round(metrics_dict[key], decimals=3).tolist()
        return metrics_dict
