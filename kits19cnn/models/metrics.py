import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def dice_plus_xent_loss(out_act="sigmoid", n_classes=3):
    """
    Function to calculate the loss used in https://arxiv.org/pdf/1809.10486.pdf,
    no-new net, Isenseee et al (used to win the Medical Imaging Decathlon).
    It is the sum of the cross-entropy and the Dice-loss.
    Args:
        out_act (str): output activation
        n_classes (int): number of classes (don't include the background class if using sigmoid)
    Return:
        the loss (cross_entropy + dice loss)
    """
    # Dice as according to the paper:
    def _loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        if out_act == "softmax":
            l_xent = K.categorical_crossentropy(y_true, y_pred, axis=1)
            l_dice = soft_dice_loss(y_true, y_pred)
        elif out_act == "sigmoid":
            # Adjusted for multi-label cases (labels must be one-hot encoded)
            l_xent = K.mean(K.binary_crossentropy(y_true, y_pred, axis=1))
            l_dice = soft_dice_loss(y_true, y_pred)
        elif out_act is None:
            # Adjusted for mutually exlusive classes and sparse labels cases
            l_xent = sparse_categorical_crossentropy_with_logits(y_true, y_pred)
            l_dice_fn = sparse_dice_loss(n_classes, from_logits=True)
            l_dice = l_dice_fn(y_true, y_pred)
        # Add the losses
        return l_dice + l_xent
    return _loss

def sparse_categorical_crossentropy_with_logits(y_true, y_pred):
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=1))

def sparse_dice_loss(n_classes, include_background=True, from_logits=False):
    """
    Dice loss for sparse labels (not one-hot encoded)
    Args:
        n_classes (int): Number of classes in image; includes the background class
    Returns:
        _loss: keras loss function
    """
    dice = sparse_dice(n_classes, include_background=include_background, from_logits=from_logits)
    def _loss(y_true, y_pred):
        return -dice(y_true, y_pred)
    return _loss

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
    axes = tuple(range(2, len(y_true.get_shape()))) # assumes channels_first
    if square_numerator:
        intersect = K.sum(y_pred * y_true, axes, keepdims=False)
    else:
        intersect = K.sum((y_pred * y_true) ** 2, axes, keepdims=False)
    if square_denom:
        denom = K.sum(y_pred ** 2 + y_true ** 2, axes, keepdims=False)
    else:
        denom = K.sum(y_pred + y_true, axes, keepdims=False)
    result = K.mean(((2 * intersect + smooth_in_numerator) / (denom + smooth)))
    return result

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

def sparse_dice(n_classes, smooth=1e-5, include_background=True, only_present=False, from_logits=False):
    """Calculates a smooth Dice coefficient loss from sparse labels.
    Args:
        n_classes (int): number of class labels to evaluate on
        smooth (float): smoothing coefficient for the loss computation
        include_background (bool): flag to include a loss on the background
            label or not
        only_present (bool): flag to include only labels present in the
            inputs or not
    Returns:
        tf.Tensor: Tensor scalar representing the loss
    """
    def _dice(y_true, y_pred):
        # Get a softmax probability of the logits predictions and a one hot
        # encoding of the labels tensor
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int32)
        onehot_labels = tf.squeeze(tf.one_hot(
            indices=y_true,
            depth=n_classes,
            axis=1,
            dtype=tf.float32,
            name='onehot_labels'), axis=2)

        # Compute the Dice similarity coefficient
        axis = [0]
        label_sum = tf.reduce_sum(onehot_labels, axis=axis, name='label_sum')
        pred_sum = tf.reduce_sum(y_pred, axis=axis, name='pred_sum')
        intersection = tf.reduce_sum(onehot_labels * y_pred, axis=axis,
                                     name='intersection')

        per_sample_per_class_dice = (2. * intersection + smooth)
        per_sample_per_class_dice /= (label_sum + pred_sum + smooth)

        # Include or exclude the background label for the computation
        if include_background:
            flat_per_sample_per_class_dice = tf.reshape(
                per_sample_per_class_dice, (-1, ))
            flat_label = tf.reshape(label_sum, (-1, ))
        else:
            flat_per_sample_per_class_dice = tf.reshape(
                per_sample_per_class_dice[:, 1:], (-1, ))
            flat_label = tf.reshape(label_sum[:, 1:], (-1, ))

        # Include or exclude non-present labels for the computation
        if only_present:
            masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                          tf.logical_not(tf.equal(flat_label, 0)))
        else:
            masked_dice = tf.boolean_mask(
                flat_per_sample_per_class_dice,
                tf.logical_not(tf.is_nan(flat_per_sample_per_class_dice)))

        dice = tf.reduce_mean(masked_dice)
        return dice
    return _dice

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
