#Resnet

from tensorflow.keras.layers import Add, Concatenate, MaxPooling2D, MaxPooling3D, \
                                    UpSampling2D, UpSampling3D, LeakyReLU, \
                                    Conv2D, Conv3D, BatchNormalization

def residual_block2D(input_layer,n_filters,strides):
    res_path = BatchNormalization(input_layer)
    res_path = LeakyReLU(0.3)(res_path)
    res_path = Conv2D(filters=n_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization(res_path)
    res_path = LeakyReLU(0.3)(res_path)
    res_path = Conv2D(filters=n_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(n_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

def encoder_2D(input_layer):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(input_layer)
    main_path = BatchNormalization(main_path)
    main_path = LeakyReLU(0.3)(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(input_layer)
    shortcut = BatchNormalization(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = residual_block2D(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = residual_block2D(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder

def decoder_2D(input_layer, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(input_layer)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = residual_block2D(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path

def residual_block3D(input_layer,n_filters,strides):
    res_path = BatchNormalization(input_layer)
    res_path = LeakyReLU(0.3)(res_path)
    res_path = Conv3D(filters=n_filters[0], kernel_size=(3, 3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = LeakyReLU(0.3)(res_path)
    res_path = Conv3D(filters=n_filters[1], kernel_size=(3, 3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv3D(n_filters[1], kernel_size=(1, 1, 1), strides=strides[0])(input_layer)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

def encoder_3D(input_layer):
    to_decoder = []

    main_path = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(input_layer)
    main_path = BatchNormalization(main_path)
    main_path = LeakyReLU(0.3)(main_path)

    main_path = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1 ,1))(main_path)

    shortcut = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1))(input_layer)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = residual_block3D(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = residual_block3D(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder

def decoder_3D(input_layer, from_encoder):
    main_path = UpSampling3D(size=(2, 2))(input_layer)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = residual_block3D(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling3D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = residual_block3D(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling3D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = residual_block3D(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def build_uresnet():
      metrics = dice_coef
    include_label_wise_dice_coefficients = True;

    inputs = Input((image_size, image_size, 1))

    to_decoder = encoder(inputs)s

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(path)

    model = Model(inputs=[inputs], outputs=[path])

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=metrics)

    return model
