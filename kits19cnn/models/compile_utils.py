from kits19cnn.models.binary_metrics import tversky_loss, focal_tversky, dice_hard, basic_bce_dice_loss
from kits19cnn.models.metrics import *
from tensorflow.keras.optimizers import Adam

def compile_softmax_version(model, opt=None, lr=1e-4, n_classes=2):
    """
    Compiling the regular softmax version of training
    """
    if opt is None:
        opt = Adam(lr=lr)
    metrics = ["accuracy"]
    loss = [dice_plus_xent_loss("softmax", n_classes)]
    model.compile(opt, loss=loss, metrics=metrics)

def compile_sparse_from_logits(model, opt=None, lr=1e-4, n_classes=3):
    """
    Compiling the sparse version of training
    """
    if opt is None:
        opt = Adam(lr=lr)
    metrics = [sparse_dice(3, from_logits=True)]
    loss = [dice_plus_xent_loss(None, n_classes)]
    model.compile(opt, loss=loss, metrics=metrics)

def compile_sigmoid_attnunet_tversky(model, lossfxn=focal_tversky, opt=None, lr=1e-4, n_classes=1):
    """
    Compiling the sigmoid+tversky version of training for RecursiveAttnUNet.
    """
    if opt is None:
        opt = Adam(lr=lr)
    # everything besides last output gets lossfxn; last output gets tversky_loss
    loss = {layer.name.split("/")[0]: (lossfxn if i!=(len(model.outputs)-1) else tversky_loss)
            for (i, layer) in enumerate(model.outputs)}
    # uniform weighting for each layer
    loss_weights = {layer.name.split("/")[0]: 1 for layer in model.outputs}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[dice_hard()])

def compile_sigmoid_attnunet(model, lossfxn=basic_bce_dice_loss, opt=None, lr=1e-4, n_classes=1):
    """
    Compiling the sigmoid+lossfxn version of training for RecursiveAttnUNet.
    """
    if opt is None:
        opt = Adam(lr=lr)
    # everything gets lossfxn
    loss = {layer.name.split("/")[0]: lossfxn for layer in model.outputs}
    # uniform weighting for each layer
    loss_weights = {layer.name.split("/")[0]: 1 for layer in model.outputs}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[dice_hard()])
