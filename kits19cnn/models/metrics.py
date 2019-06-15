# loss fn
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K

def dice_plus_xent_loss(y_true, y_pred, smooth=1e-5, out_act='sigmoid'):
    """
    Function to calculate the loss used in https://arxiv.org/pdf/1809.10486.pdf,
    no-new net, Isenseee et al (used to win the Medical Imaging Decathlon).
    It is the sum of the cross-entropy and the Dice-loss.
    Args:
        y_pred: the logits
        y_true: the one=hot encoded segmentation ground truth
    Return:
        the loss (cross_entropy + Dice)
    """
    y_pred = tf.cast(y_pred, tf.float32)
    if out_act == "softmax":
        loss_xent = K.categorical_crossentropy(y_true, y_pred)
    elif out_act == "sigmoid":
        loss_xent = K.mean(K.binary_crossentropy(y_true, y_pred), axis=1)
    # Dice as according to the paper:
    l_dice = soft_dice_loss(y_true, y_pred)
    return l_dice + l_xent

def soft_dice_loss(y_true, y_pred):
    return -soft_dice(y_true, y_pred)

def soft_dice(y_true, y_pred, smooth=1., smooth_in_numerator=1., square_numerator=False, square_denom=False):
    """
    Differentiable soft dice.
    Args:
        y_true:
        y_pred:
        smooth:
        smooth_in_numerator:
        square_numerator (bool): whether or not to the intersection of (y_true and y_pred) in the numerator
        square_denominator (bool): whether or not to square y_true and y_pred in the denominator
    Returns:
        result: the calculated soft dice
    """
    axes = tuple(range(2, len(y_true.size()))) # assumes channels_first
    if square_numerator:
        intersect = K.sum(y_pred * y_true, axes, keepdim=False)
    else:
        intersect = K.sum((y_pred * y_true) ** 2, axes, keepdim=False)
    if square_denom:
        denom = K.sum(y_pred ** 2 + y_true ** 2, axes, keepdim=False)
    else:
        denom = K.sum(y_pred + y_true, axes, keepdim=False)
    result = K.mean(((2 * intersect + smooth_in_nom) / (denom + smooth)))
    return result

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
