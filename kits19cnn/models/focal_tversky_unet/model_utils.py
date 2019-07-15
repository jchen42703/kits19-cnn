from tensorflow.keras.layers import add, multiply, Lambda, BatchNormalization, \
                                    Conv2D, Conv2DTranspose, UpSampling2D, Activation
import tensorflow.keras.backend as K

K.set_image_data_format("channels_first")  # TF dimension ordering in this code
kinit = "glorot_normal"

def expend_as(tensor, rep,name):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1), arguments={"repnum": rep},  name="psi_up"+name)(tensor)
	return my_repeat

def AttnGatingBlock(x, g, inter_shape, name):
    """ take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients"""

    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding="same", name="xl"+name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding="same")(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]), \
                                 padding="same", name="g_up"+name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation("relu")(concat_xg)
    psi = Conv2D(1, (1, 1), padding="same", name="psi"+name)(act_xg)
    sigmoid_xg = Activation("sigmoid")(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[1],  name)
    y = multiply([upsample_psi, x], name="q_attn"+name)

    result = Conv2D(shape_x[1], (1, 1), padding="same",name="q_attn_conv"+name)(y)
    result_bn = BatchNormalization(axis=1, name="q_attn_bn"+name)(result)
    return result_bn

def UnetConv2D(input, outdim, is_batchnorm, name):
	x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+"_1")(input)
	if is_batchnorm:
		x = BatchNormalization(axis=1, name=name + "_1_bn")(x)
	x = Activation("relu",name=name + "_1_act")(x)

	x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+"_2")(x)
	if is_batchnorm:
		x = BatchNormalization(axis=1, name=name + "_2_bn")(x)
	x = Activation("relu", name=name + "_2_act")(x)
	return x

def UnetGatingSignal(input, is_batchnorm, name):
    """ this is simply 1x1 convolution, bn, activation """
    shape = K.int_shape(input)
    x = Conv2D(shape[1] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + "_conv")(input)
    if is_batchnorm:
        x = BatchNormalization(axis=1, name=name + "_bn")(x)
    x = Activation("relu", name = name + "_act")(x)
    return x
