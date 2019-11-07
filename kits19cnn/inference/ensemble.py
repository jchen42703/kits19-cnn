import numpy as np

class Ensembler(object):
    """
    Iterates through multiple directories of predicted activation maps
    and averages them. The ensembled results are saved in a separate
    directory, `out_dir`.
    * Assumes the predicted activation maps are called `pred_act.npy`
    in their respective case directories.
    """
    def __init__(self):
        pass
