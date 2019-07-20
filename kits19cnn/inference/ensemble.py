import numpy as np
import os

from os.path import join, isdir
from pathlib import Path

class Ensembler(object):
    """
    Takes a list of directories of predictions (activation predictions), averages them and thresholds/softmax.
    Saves both the raw segmentations and activation maps in a separate directory.
    Main method:
        ensemble_average_predictions

    Additional features (TBD):
        1) Find optimal threshold
        2) Make a child class for ensembling the separate tumor predictions into the actual tk predictions
    """
    def __init__(self, pred_dir_list, out_dir, cases=None, regions_class_order=None):
        """
        Attributes:
            pred_dir_list: list of prediction directories
            out_dir (str): path to the directory where predictions will be saved
            cases (list): iterable of paths to case folders
            regions_class_order (tuple):
                * None for softmax on predictions
                * (tuple of label indices, excluding 0) for sigmoid on predictions
        """
        self.pred_dir_list = pred_dir_list
        self.out_dir = out_dir
        # converting cases from filepaths to raw folder names
        if cases is None:
            # Assumes that all prediction directories have the same files in them
            self.cases_raw = [case \
                              for case in os.listdir(self.pred_dir_list[-1]) \
                              if case.startswith("case")]
            assert len(self.cases) > 0, "Please make sure that the prediction directories are in the kits19/data format."
        elif cases is not None:
            # extracting raw cases from filepath cases
            cases_raw = [Path(case).name for case in cases]
            # filtering them down to only cases in the prediction directories
            self.cases_raw = [case for case in cases_raw if isdir(join(self.pred_dir_list[-1], case))]

    def ensemble_average_predictions(self):
        print("Ensembling {0}/{1}: {2}".format(i+1, len(self.cases_raw), case))
        # loading the necessary arrays
        for (i, case) in enumerate(self.cases_raw):
            save_name = "pred_{0}_act.npy".format(case)
            predictions = np.stack([np.load(join(pred_dir, case, save_name)) for pred_dir in self.pred_dir_list])
            averaged_pred = np.mean(predictions, axis=0)
            predicted_segmentation = self.convert_to_int_mask(averaged_pred)

            self.save_prediction(predicted_segmentation, averaged_pred, case)

    def convert_to_int_mask(self, act_pred):
        """
        Takes an activation map and converts it to an integer mask through argmax or thresholding.
        Args:
            act_pred (np.ndarray): activation map; shape: (n_channels, x, y, z)
        Returns:
            thresholded or argmaxed mask
        """
        if self.regions_class_order is None:
            predicted_segmentation = act_pred.argmax(0)
        else:
            predicted_segmentation_shp = act_pred[0].shape
            predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
            for i, c in enumerate(self.regions_class_order):
                predicted_segmentation[act_pred[i] > 0.5] = c

        return predicted_segmentation

    def save_prediction(self, pred, pred_act, case_raw):
        """
        Saves a prediction as a .npy array in the KiTS19 file structure
        Args:
            pred: numpy array (int)
            pred_act: numpy array (float)
            case_raw: raw case folder name (each element of self.cases_raw)
        Returns:
            None
        """
        out_case_dir = join(self.out_dir, case_raw)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)
            print("Created directory: {0}".format(out_case_dir))
        # saving the prediction and the activation map
        save_name = "pred_{0}.npy".format(case_raw)
        save_name_act = "pred_{0}_act.npy".format(case_raw)
        np.save(join(out_case_dir, save_name), pred)
        np.save(join(out_case_dir, save_name_act), pred_act)
        print("Saving: {0}, {1}".format(save_name, save_name_act))
