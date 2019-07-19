import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.losses import binary_crossentropy

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def basic_soft_dsc_loss(y_true, y_pred):
    """
    Dice loss based off of the basic dice coefficient loss.
    From "A novel focal Tversky loss function and improved Attention U-Net for lesion segmentation"
    """
    return 1 - basic_soft_dsc(y_true, y_pred)

def basic_bce_dice_loss(y_true, y_pred):
    """
    Similar to dice_plus_xent_loss, but uses the basic dice coefficient loss.
    From "A novel focal Tversky loss function and improved Attention U-Net for lesion segmentation"
    """
    loss = binary_crossentropy(y_true, y_pred) + basic_soft_dsc_loss(y_true, y_pred)
    return loss

def basic_soft_dsc(y_true, y_pred):
    """
    Basic dice coefficient. From "A novel focal Tversky loss function and improved Attention U-Net for lesion segmentation."
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_hard(threshold=0.5, axis=[1,2,3], smooth=1e-5):
    """
    Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    Args:
        threshold : float
            The threshold value to be true.
        axis : list of integer
            All dimensions are reduced, default ``[1,2,3]``.
        smooth : float
            This small value will be added to the numerator and denominator.
    """
    def dice(y_true, y_pred):
        y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
        y_true = tf.cast(y_true > threshold, dtype=tf.float32)
        inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
        hard_dice = (2. * inse + smooth) / (l + r + smooth)
        hard_dice = tf.reduce_mean(hard_dice)
        return hard_dice
    return dice

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
