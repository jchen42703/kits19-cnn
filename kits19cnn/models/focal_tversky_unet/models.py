# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:54:57 2018

@author: Nabila Abraham
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Conv2D, Conv2DTranspose, \
                                    AveragePooling2D, MaxPooling2D
import tensorflow.keras.backend as K
import kits19cnn.models.binary_metrics as metrics
from kits19cnn.models.focal_tversky_unet.model_utils import *
from functools import partial

K.set_image_data_format("channels_first")  # TF dimension ordering in this code
kinit = "glorot_normal"
concatenate = partial(concatenate, axis=1)

#model proposed in my paper - improved attention u-net with multi-scale input pyramid and deep supervision

def attn_reg(opt,input_size, lossfxn):
    img_input = Input(shape=input_size, name="input_scale1")
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name="input_scale2")(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name="input_scale3")(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name="input_scale4")(scale_img_3)

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name="conv1")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    input2 = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_scale2")(scale_img_2)
    input2 = concatenate([input2, pool1])
    conv2 = UnetConv2D(input2, 64, is_batchnorm=True, name="conv2")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    input3 = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv_scale3")(scale_img_3)
    input3 = concatenate([input3, pool2])
    conv3 = UnetConv2D(input3, 128, is_batchnorm=True, name="conv3")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    input4 = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv_scale4")(scale_img_4)
    input4 = concatenate([input4, pool3])
    conv4 = UnetConv2D(input4, 64, is_batchnorm=True, name="conv4")
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 512, is_batchnorm=True, name="center")

    g1 = UnetGatingSignal(center, is_batchnorm=True, name="g1")
    attn1 = AttnGatingBlock(conv4, g1, 128, "_1")
    up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer=kinit)(center), attn1], name="up1")

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name="g2")
    attn2 = AttnGatingBlock(conv3, g2, 64, "_2")
    up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer=kinit)(up1), attn2], name="up2")

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name="g3")
    attn3 = AttnGatingBlock(conv2, g3, 32, "_3")
    up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer=kinit)(up2), attn3], name="up3")

    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding="same", activation="relu", kernel_initializer=kinit)(up3), conv1], name="up4")

    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name="conv6")
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name="conv7")
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name="conv8")
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name="conv9")

    out6 = Conv2D(1, (1, 1), activation="sigmoid", name="pred1")(conv6)
    out7 = Conv2D(1, (1, 1), activation="sigmoid", name="pred2")(conv7)
    out8 = Conv2D(1, (1, 1), activation="sigmoid", name="pred3")(conv8)
    out9 = Conv2D(1, (1, 1), activation="sigmoid", name="final")(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

    loss = {"pred1": lossfxn,
            "pred2": lossfxn,
            "pred3": lossfxn,
            "final": metrics.tversky_loss}

    loss_weights = {"pred1": 1,
                    "pred2": 1,
                    "pred3": 1,
                    "final": 1}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[metrics.basic_soft_dsc])
    return model
