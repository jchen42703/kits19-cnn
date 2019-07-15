import tensorflow.keras.backend as K
from tensorflow.keras.layers import add, concatenate, multiply, Lambda, BatchNormalization, \
                                    Conv2D, Conv2DTranspose, UpSampling2D, Activation, MaxPooling2D
from functools import partial

K.set_image_data_format("channels_first")  # TF dimension ordering in this code
kinit = "glorot_normal"
concatenate = partial(concatenate, axis=1)

def output_module(upsamp_layer, n_filters, n_classes=1):
    """
    base_context_module_2D(pool_size=None, n_convs=2) -> conv -> sigmoid
    Args:
        upsamp_layer (tf.keras.layers.Layer): output layer from the `localization_module_2D`
        n_filters (int): number of filters for each conv layer
    Returns:
        out_seg (tf.keras.layers.Layer): output segmentation layer
    """
    conv = base_context_module_2D(upsamp_layer, n_filters, pool_size=None)
    out_seg = Conv2D(n_classes, (1, 1), activation="sigmoid")(conv)
    return out_seg

def localization_module_2D(input_layer, skip_layer, n_filters, upsamp_type="conv"):
    """
    Args:
        input_layer (tf.keras.layers.Layer): Previous depths' layers output
        skip_layer (tf.keras.layers.Layer): Layer to be skip connected to
        n_filters (int): number of filters for each conv layer
        upsamp_type (str): str describing the desired type of upsampling; "conv" or "regular";
            * "conv": The original paper implemented Conv2DTranspose w/ReLU
            * "regular": UpSampling2D -> Conv2D
    Returns:
        concat (tf.keras.layers.Layer): The concatenation of the upsampled output and the result of the AttnGatingBlock
    """
    gated = UnetGatingSignal(input_layer)
    attn1 = AttnGatingBlock(skip_layer, gated, n_filters)
    if upsamp_type.lower() == "conv":
        up = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding="same", \
                             activation="relu", kernel_initializer=kinit)(input_layer)
    elif upsamp_type.lower() == "regular":
        up = UpSampling2D((2,2))(input_layer)
        up = Conv2D(n_filters, (3,3), padding="same", activation="relu",
                    kernel_initializer=kinit)(up)
    concat = concatenate([up, attn1])
    return concat

def context_module_2D(input_layer, scale_img, n_filters, pool_size=(2,2), n_convs=2):
    """
    Args:
        input_layer:
        scale_img (tf.keras.layers.Layer): The scaled image after AveragePooling2D
        n_filters (int): number of filters for each conv layer
        n_convs (int): Number of convolutions in for the `base_context_module_2D_with_naming` function
    Returns:
        bn: Result after the final BatchNormalization layer
        pool: Result after the (2,2) MaxPooling2D layer
    """
    conv1 = Conv2D(n_filters, (3, 3), padding="same", activation="relu")(scale_img)
    input2 = concatenate([conv1, input_layer])
    bn, pool = base_context_module_2D(input2, n_filters, pool_size, n_convs=n_convs)
    return (bn, pool)

def base_context_module_2D(input_layer, n_filters, pool_size=(2, 2), n_convs=2):
    """
    [2D]; Channels_first
    Context module (Downsampling compartment of the U-Net): `n_convs` Convs (w/ ReLU and BN) -> MaxPooling
    This differs from the original papers' implementation because it places BN after the ReLU and BN
    is no longer an optional argument.
    Args:
        input_layer (tf.keras.layers.Layers):
        n_filters (int): number of filters for each conv layer
        pool_size (tuple):
        n_convs (int): Number of convolutions in the module
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        maxpooled output
    """
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            conv = Conv2D(n_filters, kernel_size=(3, 3), kernel_initializer=kinit, \
                          padding="same")(input_layer)
            act = Activation("relu")(conv)
            bn = BatchNormalization(axis=1)(act)
        else:
            conv = Conv2D(n_filters, kernel_size=(3, 3), kernel_initializer=kinit, \
                          padding="same")(bn)
            act = Activation("relu")(conv)
            bn = BatchNormalization(axis=1)(act)
    if pool_size is not None:
        pool = MaxPooling2D(pool_size)(bn)
        return (bn, pool)
    elif pool_size is None:
        return bn

def AttnGatingBlock(x, g, n_filters):
    """
    take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients
    """
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(n_filters, (2, 2), strides=(2, 2), padding="same")(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(n_filters, (1, 1), padding="same")(g)
    upsample_g = Conv2DTranspose(n_filters, (3, 3), strides=(shape_theta_x[2] // shape_g[2], \
                                 shape_theta_x[3] // shape_g[3]), padding="same")(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation("relu")(concat_xg)
    psi = Conv2D(1, (1, 1), padding="same")(act_xg)
    sigmoid_xg = Activation("sigmoid")(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[1])
    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[1], (1, 1), padding="same")(y)
    result_bn = BatchNormalization(axis=1)(result)
    return result_bn

def expend_as(tensor, rep):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1),
                       arguments={"repnum": rep})(tensor)
	return my_repeat

def UnetGatingSignal(input_layer):
    """
    This is simply 1x1 convolution, activation, bn. This differs from the original
    implementation where BN was before activation.
    Args:
        input_layer (tf.keras.layers.Layer): regular pooled/upsampled input
    Returns:
        bn (tf.keras.layers.Layer): result after applying the (1,1) Conv, ReLU, BN
    """
    n_channels = K.int_shape(input_layer)[1] # assumes channels_first
    conv = Conv2D(n_channels, (1, 1), padding="same", kernel_initializer=kinit)(input_layer)
    act = Activation("relu")(conv)
    bn = BatchNormalization(axis=1)(act)
    return bn
