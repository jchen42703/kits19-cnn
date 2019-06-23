import unittest
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from kits19cnn.models.unet.models import UNet
from kits19cnn.models.metrics import *

class TrainTest(unittest.TestCase):
    """
    Simple test case to make sure that models can train through at least one epoch
    in tested conditions. It doesn't test anything about the nature of the training
    process besides: can it run?
    """
    def setUp(self):
        self.input_shape = (5, 1, 224, 224)
        self.pred_shape = (5, 3, 224, 224)
        self.x = np.random.normal(0, 1, self.input_shape).astype(np.float32)
        self.y = np.random.randint(0, 2, self.input_shape).astype(np.int32) # blank label
        self.pred = np.random.normal(0, 1, self.pred_shape).astype(np.float32)
        self.y_reg = None # regular label
        self.filters0 = 5

    def test_train_sparse_xent_and_sparse_dice(self):
        n_classes = 3
        u_model = UNet(self.input_shape[-3:], n_classes=n_classes, n_pools=5, starting_filters=self.filters0)
        model = u_model.build_model(include_top=True, out_act=None)
        def _compile_xent_dice_combined(model, lr, opt=None, n_classes=3):
            """
            dice loss + xent loss combined using keras.models.Model.compile.
            """
            if opt is None:
                opt = Adam(lr=lr)
            metrics = ["accuracy"]
            _loss = [dice_plus_xent_loss(out_act=None, n_classes=n_classes)]
            model.compile(opt, loss=_loss, metrics=metrics)
        _compile_xent_dice_combined(model, lr=1e-4, n_classes=n_classes)
        hist = model.fit(self.x, self.y, batch_size=1, epochs=1)
        print("Ran: {0}".format("test_train_sparse_xent_and_sparse_dice"))
        self.assertTrue(True)

    def test_train_sparse_xent_and_sparse_dice_no_include_background(self):
        n_classes = 3
        u_model = UNet(self.input_shape[-3:], n_classes=n_classes, n_pools=5, starting_filters=self.filters0)
        model = u_model.build_model(include_top=True, out_act=None)
        def _compile_xent_dice_combined(model, lr, opt=None, n_classes=3):
            """
            dice loss + xent loss combined using keras.models.Model.compile.
            """
            if opt is None:
                opt = Adam(lr=lr)
            metrics = ["accuracy"]
            def _loss(y_true, y_pred):
                l_dice_fn = sparse_dice_loss(n_classes, include_background=False, from_logits=True)
                l_dice = l_dice_fn(y_true, y_pred)
                l_xent = sparse_categorical_crossentropy_with_logits(y_true, y_pred)
                return l_dice + l_xent
            model.compile(opt, loss=_loss, metrics=metrics)
        _compile_xent_dice_combined(model, lr=1e-4, n_classes=n_classes)
        hist = model.fit(self.x, self.y, batch_size=1, epochs=1)
        print("Ran: {0}".format("test_train_sparse_xent_and_sparse_dice_no_include_background"))
        self.assertTrue(True)

if __name__ == "__main__":
    K.set_image_data_format("channels_first")
    unittest.main()
