from tensorflow.keras.layers import Add, Concatenate, MaxPooling2D, MaxPooling3D, \
                                    UpSampling2D, UpSampling3D, LeakyReLU, \
                                    Conv2D, Conv3D, BatchNormalization

def context_module(input_layer, n_filters, pool_size=(2,2,2), n_convs=2):
    """
    [3D]; Channels_first
    Context module (Downsampling compartment of the U-Net): `n_convs` Convs (w/ LeakyReLU and BN) -> MaxPooling
    Args:
        input_layer (tf.keras layer):
        n_filters (int): number of filters for each conv layer
        pool_size (tuple):
        n_convs (int): Number of convolutions in the module
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        maxpooled output
    """
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            conv = Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(input_layer)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
        else:
            conv = Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(bn)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
    if pool_size is not None:
        pool = MaxPooling3D(pool_size)(bn)
        return bn, pool
    elif pool_size is None:
        return bn

def context_module_2D(input_layer, n_filters, pool_size=(2,2), n_convs=2):
    """
    [2D]; Channels_first
    Context module (Downsampling compartment of the U-Net): `n_convs` Convs (w/ LeakyReLU and BN) -> MaxPooling
    Args:
        input_layer (tf.keras layer):
        n_filters (int): number of filters for each conv layer
        pool_size (tuple):
        n_convs (int): Number of convolutions in the module
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        maxpooled output
    """
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            conv = Conv2D(n_filters, kernel_size=(3,3), padding='same')(input_layer)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
        else:
            conv = Conv2D(n_filters, kernel_size=(3,3), padding='same')(bn)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
    if pool_size is not None:
        pool = MaxPooling2D(pool_size)(bn)
        return bn, pool
    elif pool_size is None:
        return bn

def localization_module(input_layer, skip_layer, n_filters, upsampling_size=(2,2,2), n_convs=2):
    """
    [3D]; Channels_first
    Localization module (Downsampling compartment of the U-Net): UpSampling3D -> `n_convs` Convs (w/ LeakyReLU and BN)
    Args:
        input_layer (tf.keras layer):
        skip_layer (tf.keras layer): layer with the corresponding skip connection (same depth)
        n_filters (int): number of filters for each conv layer
        upsampling_size (tuple):
        n_convs (int): Number of convolutions in the module
    Returns:
        upsampled output
    """
    upsamp = UpSampling3D(upsampling_size)(input_layer)
    concat = Concatenate(axis=1)([upsamp, skip_layer])
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            conv = Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(concat)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
        else:
            conv = Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(bn)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
    return bn

def localization_module_2D(input_layer, skip_layer, n_filters, upsampling_size=(2,2), n_convs=2):
    """
    [2D]; Channels_first
    Localization module (Downsampling compartment of the U-Net): UpSampling2D -> `n_convs` Convs (w/ LeakyReLU and BN)
    Args:
        input_layer (tf.keras layer):
        skip_layer (tf.keras layer): layer with the corresponding skip connection (same depth)
        n_filters (int): number of filters for each conv layer
        upsampling_size (tuple):
        n_convs (int): Number of convolutions in the module
    Returns:
        upsampled output
    """
    upsamp = UpSampling2D(upsampling_size)(input_layer)
    concat = Concatenate(axis=1)([upsamp, skip_layer])
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            conv = Conv2D(n_filters, kernel_size=(3,3), padding='same')(concat)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
        else:
            conv = Conv2D(n_filters, kernel_size=(3,3), padding='same')(bn)
            act = LeakyReLU(0.3)(conv)
            bn = BatchNormalization(axis=1)(act)
    return bn
