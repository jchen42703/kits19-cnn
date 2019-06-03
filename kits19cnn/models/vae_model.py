# Keras implementation of the paper:
# 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization
# by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
# Author of this code: Suyog Jadhav (https://github.com/IAmSUyogJadhav)

import keras.backend as K
from keras.losses import mse
from keras.layers import Conv3D, Activation, Add, UpSampling3D, Lambda, Dense
from keras.layers import Input, Reshape, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from kits19cnn.models.utils.vae_utils import *

def build_model(input_shape=(4, 160, 192, 128), n_classes=3, lr=1e-4):
    """
    build_model(input_shape=(4, 160, 192, 128))
    -------------------------------------------
    Creates the model used in the BRATS2018 winning solution
    by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
    Parameters
    ----------
    `input_shape`: A 4-tuple, optional.
        Shape of the input image. Must be a 4D image of shape (c, H, W, D),
        where, each of H, W and D are divisible by 2^4. Defaults to the crop
        size used in the paper, i.e., (4, 160, 192, 128).
    Returns
    -------
    `model`: A keras.models.Model instance
        The created model.
    """
    c, H, W, D = input_shape
    assert len(input_shape) == 4, "Input shape must be a 4-tuple"
    assert ~(H % 16) and ~(W % 16) and ~(D % 16), \
        "All the input dimensions must be divisible by 16"

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------

    ## Input Layer
    inp = Input(input_shape)

    ## The Initial Block
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1,padding='same', data_format='channels_first', name='Input_x1')(inp)

    ## Dropout (0.2)
    x = Dropout(0.2)(x)

    ## Green Block x1 (output filters = 32)
    x1 = green_block(x, 32, name='x1')
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=2, padding='same', data_format='channels_first', name='Enc_DownSample_32')(x1)

    ## Green Block x2 (output filters = 64)
    x = green_block(x, 64, name='Enc_64_1')
    x2 = green_block(x, 64, name='x2')
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=2, padding='same', data_format='channels_first', name='Enc_DownSample_64')(x2)

    ## Green Blocks x2 (output filters = 128)
    x = green_block(x, 128, name='Enc_128_1')
    x3 = green_block(x, 128, name='x3')
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=2, padding='same', data_format='channels_first', name='Enc_DownSample_128')(x3)

    ## Green Blocks x4 (output filters = 256)
    x = green_block(x, 256, name='Enc_256_1')
    x = green_block(x, 256, name='Enc_256_2')
    x = green_block(x, 256, name='Enc_256_3')
    x4 = green_block(x, 256, name='x4')

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    ## GT (Groud Truth) Part
    # -------------------------------------------------------------------------

    ### Green Block x1 (output filters=128)
    x = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_GT_ReduceDepth_128')(x4)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_128')(x)
    x = Add(name='Input_Dec_GT_128')([x, x3])
    x = green_block(x, 128, name='Dec_GT_128')

    ### Green Block x1 (output filters=64)
    x = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_GT_ReduceDepth_64')(x)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_64')(x)
    x = Add(name='Input_Dec_GT_64')([x, x2])
    x = green_block(x, 64, name='Dec_GT_64')

    ### Green Block x1 (output filters=32)
    x = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_GT_ReduceDepth_32')(x)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_GT_UpSample_32')(x)
    x = Add(name='Input_Dec_GT_32')([x, x1])
    x = green_block(x, 32, name='Dec_GT_32')

    ### Blue Block x1 (output filters=32)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', data_format='channels_first', name='Input_Dec_GT_Output')(x)

    ### Output Block
    out_GT = Conv3D(filters=n_classes, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', activation='sigmoid', name='Dec_GT_Output')(x)

    ## VAE (Variational Auto Encoder) Part
    # -------------------------------------------------------------------------
    ### VD Block (Reducing dimensionality of the data)
    x = GroupNormalization(groups=8, axis=1, name='Dec_VAE_VD_GN')(x4)
    x = Activation('relu', name='Dec_VAE_VD_relu')(x)
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=2, padding='same', data_format='channels_first', name='Dec_VAE_VD_Conv3D')(x)

    # Not mentioned in the paper, but the author used a Flattening layer here.
    x = Flatten(name='Dec_VAE_VD_Flatten')(x)
    x = Dense(256, name='Dec_VAE_VD_Dense')(x)

    ### VDraw Block (Sampling)
    z_mean = Dense(128, name='Dec_VAE_VDraw_Mean')(x)
    log_z_var = Dense(128, name='Dec_VAE_VDraw_Var')(x)
    x = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')([z_mean, log_z_var])

    ### VU Block (Upsizing back to a depth of 256)
    low_h, low_w, low_d = input_shape[1]//16, input_shape[2]//16, input_shape[3]//16
    x = Dense(1 * low_h * low_w * low_d)(x)
    x = Activation('relu')(x)
    x = Reshape((1, low_h, low_w, low_d))(x)
    x = Conv3D(filters=256, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_VAE_ReduceDepth_256')(x)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_256')(x)

    ### Green Block x1 (output filters=128)
    x = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_VAE_ReduceDepth_128')(x)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_128')(x)
    x = green_block(x, 128, name='Dec_VAE_128')

    ### Green Block x1 (output filters=64)
    x = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_VAE_ReduceDepth_64')(x)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_64')(x)
    x = green_block(x, 64, name='Dec_VAE_64')

    ### Green Block x1 (output filters=32)
    x = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_VAE_ReduceDepth_32')(x)
    x = UpSampling3D(size=2, data_format='channels_first', name='Dec_VAE_UpSample_32')(x)
    x = green_block(x, 32, name='Dec_VAE_32')

    ### Blue Block x1 (output filters=32)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', data_format='channels_first', name='Input_Dec_VAE_Output')(x)

    ### Output Block
    out_VAE = Conv3D(filters=int(inp.shape[1]), kernel_size=(1, 1, 1), strides=1, data_format='channels_first', name='Dec_VAE_Output')(x)
    # Build and Compile the model
    out = out_GT
    model = Model(inp, out)  # Create the model
    model.compile(
        Adam(lr=lr),
        loss(input_shape, inp, out_VAE, z_mean, log_z_var),
        metrics=['accuracy']
            )

    return model
