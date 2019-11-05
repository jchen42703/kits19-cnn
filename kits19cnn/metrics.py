import numpy as np

def evaluate_official(y_true, y_pred):
    """
    Official evaluation metric. (numpy)
    """
    try:
        # Compute tumor+kidney Dice
        tk_pd = np.greater(y_pred, 0)
        tk_gt = np.greater(y_true, 0)
        intersection = np.logical_and(tk_pd, tk_gt).sum()
        tk_dice = 2*intersection/(tk_pd.sum() + tk_gt.sum())
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = np.greater(y_pred, 1)
        tu_gt = np.greater(y_true, 1)
        intersection = np.logical_and(tu_pd, tu_gt).sum()
        tu_dice = 2*intersection/(tu_pd.sum() + tu_gt.sum())
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice, tu_dice
