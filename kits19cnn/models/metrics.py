# loss fn
import keras
import tensorflow as tf
import keras.backend as K
def dice_plus_xent_loss(y_true, y_pred, smooth = 1e-5, out_act = 'sigmoid'):
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
    multi_class = y_pred.shape[-1] >= 2
    if out_act == "softmax":
        loss_xent = K.mean(K.categorical_crossentropy(y_true, y_pred))
    elif out_act == "sigmoid":
#         assert y_pred.shape[-1] == 1, "Please check that your outputted segmenatations are single channel \
#                                        for binary cross entropy."
        loss_xent = K.mean(K.binary_crossentropy(y_true, y_pred), axis = -1)
    # Dice as according to the paper:
    n_dims = len(y_pred.shape) - 2 # subtracting the batch_size and n_channels dimensions
    axes = [axis for axis in range(n_dims + 1)] # to sum over all dimensions besides the channels (classes)
    # axes = [0]
    numerator = 2.0 * K.sum(y_true * y_pred, axis = axes)
    denominator = K.sum(y_pred, axes) + K.sum(y_true, axes)
    if multi_class:
        loss_dice = K.mean((numerator + smooth) / (denominator + smooth))
    elif not multi_class:
        loss_dice = (numerator + smooth) / (denominator + smooth)
    l_dice = tf.Print(loss_dice, [loss_dice], "dice: ")
    l_xent = tf.Print(loss_xent, [loss_xent], "xent: ")
#     l_xent = K.print_tensor(loss_xent, message = 'xent')
#     l_dice = K.print_tensor(loss_dice, message = 'multi-class dice')
#     pr_fn = tf.Print([loss_dice, loss_xent], [loss_dice, loss_xent], "soft dice: " + str(eval(loss_dice)) + "xent: " + str(eval(loss_xent)))
    return -l_dice + l_xent

def multi_class_dice(y_true, y_pred, smooth = 1e-5):
    """
    Simple multi-class dice coefficient that computes the average dice for each class.
    This implementation assumes a "channels_last" tensor format.
    Args:
        y_true:
        y_pred:
        smooth: small value to avoid division by 0
            default: 1e-5
    Returns:
        The mean dice coefficient over all of the classes.
    """
    n_dims = len(y_pred.shape) - 2 # subtracting the batch_size and n_channels dimensions
    axes = [axis for axis in range(n_dims + 1)] # to sum over all dimensions besides the channels (classes)
    intersect = K.sum(y_true * y_pred, axis=axes)
    numerator = 2 * intersect + smooth
    denominator = K.sum(y_true, axis = axes) + K.sum(y_pred, axis = axes) + smooth
    return K.mean(numerator / denominator)

def multi_class_dice_loss(y_true, y_pred):
    return -multi_class_dice(y_true, y_pred)
