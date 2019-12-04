import segmentation_models_pytorch as smp
import torch

from kits19cnn.utils import softmax_helper
from kits19cnn.models import Generic_UNet, wrap_smp_model
from .utils import get_preprocessing
from kits19cnn.io import TestVoxelDataset
from .infer import BaseInferenceExperiment

class SegmentationInferenceExperiment2D(BaseInferenceExperiment):
    """
    Inference Experiment to support prediction experiments
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict):
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_datasets(self, test_ids):
        """
        Creates and returns the test dataset.
        """
        use_rgb = "smp" in self.model_params["architecture"]
        preprocess_t = get_preprocessing(use_rgb)
        # creating the datasets
        test_dataset = TestVoxelDataset(im_ids=test_ids,
                                        transforms=None,
                                        preprocessing=preprocess_t,
                                        file_ending=self.io_params["file_ending"])
        return test_dataset

    def get_model(self):
        """
        Fetches the 2D model: the nnU-Net, smp U-Net or smp FPN for prediction.
        """
        architecture = self.model_params["architecture"]
        # creating model
        if architecture == "nnunet":
            unet_kwargs = self.model_params[architecture]
            unet_kwargs = self.setup_2D_UNet_params(unet_kwargs)
            model = Generic_UNet(**unet_kwargs)
            model.inference_apply_nonlin = softmax_helper
        # smp models
        elif "smp" in architecture:
            model_type = smp.FPN if architecture == "fpn_smp" else smp.Unet
            print(f"Model type: {model_type}")
            model_kwargs = {"encoder_name": self.model_params["encoder"],
                            "encoder_weights": None, "classes": 3,
                            "activation": None}
            model_kwargs.update(self.model_params[architecture])
            # adds the `predict_3D` method for the smp model
            model = wrap_smp_model(model_type, model_kwargs,
                                   num_classes=model_kwargs["classes"],
                                   activation=self.model_params["activation"])
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model.cuda()

    def setup_2D_UNet_params(self, unet_kwargs):
        """
        ^^^^^^^^^^^^^
        """
        unet_kwargs["conv_op"] = torch.nn.Conv2d
        if self.model_params.get("instance_norm"):
            unet_kwargs["norm_op"] = torch.nn.InstanceNorm2d
        unet_kwargs["dropout_op"] = torch.nn.Dropout2d
        unet_kwargs["nonlin"] = torch.nn.ReLU
        unet_kwargs["nonlin_kwargs"] = {"inplace": True}
        unet_kwargs["final_nonlin"] = lambda x: x
        return unet_kwargs
