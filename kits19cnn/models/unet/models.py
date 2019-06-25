from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Conv3D
from tensorflow.keras import backend as K
from functools import partial
from kits19cnn.models.unet import model_utils

class UNet(object):
    """
    Regular U-Net; 2D/3D
    Attributes:
        input_shape (tuple, list): (n_channels, x, y(,z))
        n_convs (int): number of convs per block
        n_classes (int): number of output classes
        n_pools (int): Number of max pooling operations (depth = n_pools+1)
        starting_filters (int): # of filters at the highest layer (first convs)
    """
    def __init__(self, input_shape=(1, None, None, None), n_convs=2, n_classes=1,
                 n_pools=6, starting_filters=30):
        self.input_shape = input_shape
        self.n_pools = n_pools
        self.ndim = len(input_shape[1:]) # not including channels dimension
        self.n_classes = n_classes
        self.pool_size = tuple([2 for i in range(self.ndim)])
        if self.ndim == 2:
            self.context_mod = partial(model_utils.context_module_2D, n_convs=n_convs)
            self.localization_mod = partial(model_utils.localization_module_2D, n_convs=n_convs)
        elif self.ndim == 3:
            self.context_mod = partial(model_utils.context_module, n_convs=n_convs)
            self.localization_mod = partial(model_utils.localization_module, n_convs=n_convs)
        self.filter_list = [starting_filters*(2**level) for level in range(0, n_pools+1)]

    def build_model(self, include_top=False, input_layer=None, out_act='sigmoid'):
        """
        Returns a keras.models.Model instance.
        Args:
            input_shape: shape w/o batch_size and n_channels; must be a tuple of ints with length 3
            include_top (bool): final conv + activation layers to produce segmentation
            out_act (str): Output activation; either "sigmoid", "softmax" or None
        Returns:
            keras.models.Model
        """
        input_layer = Input(shape=self.input_shape)
        skip_layers = []
        level = 0
        # context pathway (downsampling) [level 0 to (depth - 1)]
        while level < self.n_pools:
            if level == 0:
                skip, pool = self.context_mod(input_layer, self.filter_list[level], pool_size=self.pool_size)
            elif level > 0:
                skip, pool = self.context_mod(pool, self.filter_list[level], pool_size=self.pool_size)
            skip_layers.append(skip)
            level += 1
        convs_bottom = self.context_mod(pool, self.filter_list[level], pool_size=None) # No downsampling;  level at (depth) after the loop
        # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level-1
            if level == self.n_pools:
                upsamp = self.localization_mod(convs_bottom, skip_layers[current_depth], self.filter_list[current_depth], upsampling_size=self.pool_size)
            elif not level == self.n_pools:
                upsamp = self.localization_mod(upsamp, skip_layers[current_depth], self.filter_list[current_depth], upsampling_size=self.pool_size)
            level -= 1
        # output convolutions
        if self.ndim == 2:
            conv_seg = Conv2D(self.n_classes, kernel_size=(1,1), activation=out_act)(upsamp)
        elif self.ndim == 3:
            conv_seg = Conv3D(self.n_classes, kernel_size=(1,1,1), activation=out_act)(upsamp)
        # return feature maps
        if not include_top:
            extractor = Model(inputs=[input_layer], outputs=[upsamp])
            return extractor
        # return the segmentation
        elif include_top:
            unet = Model(inputs=[input_layer], outputs=[conv_seg])
            return unet
        return unet
