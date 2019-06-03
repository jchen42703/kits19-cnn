# loss fn
import keras
import tensorflow as tf
import keras.backend as K

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
