import torch
from kits19cnn.models.nnunet.neural_network import SegmentationNetwork

def wrap_smp_model(smp_model_type, smp_model_kwargs={}):
    """
    Wraps a 2D smp model with SegmentationNetwork's methods. Mainly for
    inference, so that the smp_model can use the `predict_3D` method.
    """
    class WrappedModel(smp_model_type, SegmentationNetwork):
        def __init__(self, model_kwargs={}):
            super().__init__(**model_kwargs)
            self.conv_op = torch.nn.Conv2d

    wrapped_model = WrappedModel(smp_model_kwargs)
    return wrapped_model
