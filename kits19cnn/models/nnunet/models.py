from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D
from tensorflow.keras import backend as K
from functools import partial
from kits19cnn.models.nnunet import model_utils

class AdaptiveUNet(model_utils.AdaptiveNetwork):
    """
    Isensee's 2D/3D U-Net for Heart Segmentation from the MSD that follows the conditions:
        * pools until the feature maps axes are all of at <= 8
        * max # of pools = 6 for 2D
    Augmented to allow for use as a feature extractor
    Attributes:
        input_shape: The shape of the input including the number of input channels; (n_channels, z, x, y)
        n_convs: number of convolutions per module
        n_classes: number of output classes (default: 1, which is binary segmentation)
            * Make sure that it doesn't include the background class (0)
        max_pools: max number of max pooling layers
        starting_filters: number of filters at the highest depth
    """
    def __init__(self, input_shape, n_convs=2, n_classes=1, max_pools=6, starting_filters=30,):
        super().__init__(input_shape, max_pools, starting_filters, base_pool_size=2)
        self.n_convs = n_convs
        self.n_classes = n_classes
        if self.ndim == 2:
            self.context_mod = partial(model_utils.context_module_2D, n_convs=n_convs)
            self.localization_mod = partial(model_utils.localization_module_2D, n_convs=n_convs)
        elif self.ndim == 3:
            self.context_mod = partial(model_utils.context_module, n_convs=n_convs)
            self.localization_mod = partial(model_utils.localization_module, n_convs=n_convs)
        # automatically reassigns the max number of pools in a model (for cases where the actual max pools < inputted one)
        self.max_pools = max(self._pool_statistics())

    def build_model(self, include_top=False, input_layer=None, out_act='sigmoid'):
        """
        Returns a keras.models.Model instance.
        Args:
            include_top (boolean): Whether or not you want to have a segmentation layer
            input_layer: keras layer
                * if None, then defaults to a regular input layer based on the shape
            extractor: boolean on whether or not to use the U-Net as a feature extractor or not
            out_act: string representing the activation function for the last layer. Should be either "softmax" or "sigmoid".
            Defaults to "sigmoid".
        """
        if input_layer is None:
            input_layer = Input(shape = self.input_shape)
        skip_layers = []
        level = 0
        # context pathway (downsampling) [level 0 to (depth - 1)]
        while level < self.max_pools:
            if level == 0:
                skip, pool = self.context_mod(input_layer, self.filter_list[level], pool_size = self.pool_list[0])
            elif level > 0:
                skip, pool = self.context_mod(pool, self.filter_list[level], pool_size = self.pool_list[level])
            skip_layers.append(skip)
            level += 1
        convs_bottom = self.context_mod(pool, self.filter_list[level], pool_size = None) # No downsampling;  level at (depth) after the loop
        # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level-1
            if level == self.max_pools:
                upsamp = self.localization_mod(convs_bottom, skip_layers[current_depth], self.filter_list[current_depth],\
                                               upsampling_size = self.pool_list[current_depth])
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_layers[current_depth], self.filter_list[current_depth],\
                                               upsampling_size = self.pool_list[current_depth])
            level -= 1

        if self.ndim == 2:
            conv_seg = Conv2D(self.n_classes, kernel_size=(1,1), activation=out_act)(upsamp)
        elif self.ndim == 3:
            conv_seg = Conv3D(self.n_classes, kernel_size=(1,1,1), activation=out_act)(upsamp)
        # return feature maps
        if not include_top:
            extractor = Model(inputs = [input_layer], outputs = [upsamp])
            return extractor
        # return the segmentation
        elif include_top:
            unet = Model(inputs = [input_layer], outputs = [conv_seg])
            return unet

class UNet(object):
    """
    Regular U-Net; 2D/3D
    Attributes:
        n_convs:
        depth: max # of pools + 1
        n_channels:
        starting_filters:
    """
    def __init__(self, input_shape=(1, None, None, None), n_convs=2, n_classes=1, max_pools=6,
                 starting_filters=30,):
        self.input_shape = input_shape
        self.max_pools = max_pools
        self.ndim = len(input_shape[1:]) # not including channels dimension
        self.n_classes = n_classes
        self.pool_size = tuple([2 for i in range(self.ndim)])
        if self.ndim == 2:
            self.context_mod = partial(model_utils.context_module_2D, n_convs=n_convs)
            self.localization_mod = partial(model_utils.localization_module_2D, n_convs=n_convs)
        elif self.ndim == 3:
            self.context_mod = partial(model_utils.context_module, n_convs=n_convs)
            self.localization_mod = partial(model_utils.localization_module, n_convs=n_convs)
        self.filter_list = [starting_filters*(2**level) for level in range(0, max_pools+1)]

    def build_model(self, include_top=False, input_layer=None, out_act='sigmoid'):
        """
        Returns a keras.models.Model instance.
        Args:
            input_shape: shape w/o batch_size and n_channels; must be a tuple of ints with length 3
        """
        input_layer = Input(shape=self.input_shape)
        skip_layers = []
        level = 0
        # context pathway (downsampling) [level 0 to (depth - 1)]
        while level < self.max_pools:
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
            if level == self.max_pools:
                upsamp = self.localization_mod(convs_bottom, skip_layers[current_depth], self.filter_list[current_depth], upsampling_size=self.pool_size)
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_layers[current_depth], self.filter_list[current_depth], upsampling_size=self.pool_size)
            level -= 1

        if self.ndim == 2:
            conv_seg = Conv2D(self.n_classes, kernel_size=(1,1), activation=out_act)(upsamp)
        elif self.ndim == 3:
            conv_seg = Conv3D(self.n_classes, kernel_size=(1,1,1), activation=out_act)(upsamp)
        # return feature maps
        if not include_top:
            extractor = Model(inputs = [input_layer], outputs = [upsamp])
            return extractor
        # return the segmentation
        elif include_top:
            unet = Model(inputs = [input_layer], outputs = [conv_seg])
            return unet
        return unet
