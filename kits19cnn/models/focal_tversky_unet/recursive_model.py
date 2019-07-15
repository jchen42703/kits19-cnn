from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Conv2D, Conv2DTranspose, \
                                    AveragePooling2D, MaxPooling2D
import tensorflow.keras.backend as K
from kits19cnn.models.focal_tversky_unet.recursive_utils import *
from functools import partial

K.set_image_data_format("channels_first")  # TF dimension ordering in this code
kinit = "glorot_normal"
concatenate = partial(concatenate, axis=1)

class RecursiveAttenUNet(object):
    """
    Recursive version of the Attention Gated U-Net; 2D only
    Attributes:
        input_shape (tuple, list): (n_channels, x, y(,z))
        n_classes (int): number of output classes
        n_pools (int): Number of max pooling operations (depth = n_pools+1)
        filter_list (list): list of filters (len: n_pools+1)
        starting_filters (int): # of filters at the highest layer (first convs)
        upsamp_type (str): str describing the desired type of upsampling; "conv" or "regular";
            * "conv": The original paper implemented Conv2DTranspose w/ReLU
            * "regular": UpSampling2D -> Conv2D
    """
    def __init__(self, input_shape=(1, None, None), n_classes=1, n_pools=6,
                 filter_list=None, starting_filters=None, upsamp_type="regular"):
        self.input_shape = input_shape
        self.n_pools = n_pools
        self.n_classes = n_classes
        self.pool_size = (2,2)
        if filter_list is None:
            if starting_filters is None:
                starting_filters = 32
            self.filter_list = [starting_filters*(2**level) for level in range(0, n_pools+1)]
        self.upsamp_type = upsamp_type

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
        ### Comments refer to level=0 as the top section of the U-Net (input-output)
        input_layer = Input(shape=self.input_shape)
        # initializing all scaled input (occur from level=1 to level=n_pools-1)
        ## Each of these scaled images will go through convs to an eventual AG
        scale_imgs = self.construct_scaled_inputs(input_layer)
        # context pathway (downsampling) [level 0 to (depth - 1)]
        skip_layers, pool = self.construct_context_pathway(input_layer, scale_imgs)
        convs_bottom = base_context_module_2D(pool, self.filter_list[-1], pool_size=None) # No downsampling;  level at (depth) after the loop
        # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
        upsamp_list = self.construct_localization_pathway(convs_bottom, skip_layers)
        # output convolutions
        ## Applying the sigmoid to each of the gated upsampled outputs
        outputs_list = [output_module(upsamp_list[n_layer], self.filter_list[-n_layer-1], self.n_classes)\
                        for n_layer in range(len(upsamp_list))]
        # return the segmentation
        unet = Model(inputs=[input_layer], outputs=outputs_list)
        return unet

    def construct_scaled_inputs(self, img_input_layer):
        """
        Initializing all scaled input (occur from level=1 to level=n_pools-1).
        Note that level=n_pools-1 is the level right before the bottom convs.
        Each of these scaled images will go through convs to an eventual AG.
        The scaling is done through average pooling.

        Args:
            img_input_layer: the first input layer
        Returns:
            list of all of the scaled image layers excluding the original input layer
        """
        scale_imgs = []
        for pool in range(self.n_pools-1):
            if pool == 0:
                scale_imgs.append(AveragePooling2D(pool_size=(2, 2))(img_input_layer))
            else:
                # average pools previous average pooled image
                scale_imgs.append(AveragePooling2D(pool_size=(2, 2))(scale_imgs[pool-1]))
        return scale_imgs

    def construct_context_pathway(self, input_layer, scale_img_list):
        """
        Constructs the context downsampling pathway from level=0 to level=depth-1.

        Args:
            input_layer: original input layer
            scale_img_list: list of all of the scaled averaged pooled image layers
                * This is the output from self.construct_scaled_inputs
        Returns:
            skip_layers: list of layers to be skip connected (the last layer in each level
            before pooling)
            pool: the pooling layer
        """
        # context pathway (downsampling) [level 0 to (depth - 1)]
        skip_layers = []
        level = 0
        while level < self.n_pools:
            if level == 0:
                skip, pool = base_context_module_2D(input_layer, self.filter_list[level], pool_size=self.pool_size)
            elif level > 0:
                skip, pool = context_module_2D(pool, scale_img_list[level-1], self.filter_list[level], \
                                               pool_size=self.pool_size)
            skip_layers.append(skip)
            level += 1
        return skip_layers, pool

    def construct_localization_pathway(self, convs_bottom_layer, skip_layers_list):
        """
        Constructs the upsampling localization pathway. This deals with the attention gating
        as well as the skip connections. From level=depth-1 to level=1.

        Args:
            conv_bottom_layer: output layer at level=depth (bottom-most)
            skip_layers_list: list of layers that are to be fed through an AG
            and then skip-connected
        Returns:
            upsamp_layers: list of upsampled layers at each level (to eventually
            be evaluated on for deep supervision)
        """
        # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
        upsamp_layers = []
        level = self.n_pools # aka ** level = depth - 1 at the start of the loop)
        while level > 0:
            # Remember: depth=n_pools+1, so when level==n_pools, this is one layer above the bottom block
            if level == self.n_pools:
                upsamp = localization_module_2D(convs_bottom_layer, skip_layers_list[level-1], \
                                               self.filter_list[level-1], self.upsamp_type)
                upsamp_layers.append(upsamp)
            elif not level == self.n_pools:
                upsamp = localization_module_2D(upsamp, skip_layers_list[level-1], \
                                               self.filter_list[level-1], self.upsamp_type)
                upsamp_layers.append(upsamp)
            level -= 1
        return upsamp_layers
