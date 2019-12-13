from catalyst.dl.runner import SupervisedRunner

from kits19cnn.experiments import TrainSegExperiment, TrainClfSegExperiment3D, \
                                  TrainSegExperiment2D, TrainClfSegExperiment2D, \
                                  seed_everything
from kits19cnn.visualize import plot_metrics, save_figs

def main(config):
    """
    Main code for training a classification/seg/classification+seg model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. script/configs/train.yml
    Returns:
        None
    """
    # setting up the train/val split with filenames
    seed = config["io_params"]["split_seed"]
    seed_everything(seed)
    mode = config["mode"].lower()
    assert mode in ["classification", "segmentation", "both"], \
        "The `mode` must be one of ['classification', 'segmentation', 'both']."
    if mode == "classification":
        raise NotImplementedError
    elif mode == "segmentation":
        if config["dim"] == 2:
            exp = TrainSegExperiment2D(config)
        elif config["dim"] == 3:
            exp = TrainSegExperiment(config)
        output_key = "logits"
    elif mode == "both":
        if config["dim"] == 2:
            exp = TrainClfSegExperiment2D(config)
        elif config["dim"] == 3:
            exp = TrainClfSegExperiment3D(config)
        output_key = ["seg_logits", "clf_logits"]

    print(f"Seed: {seed}\nMode: {mode}")

    runner = SupervisedRunner(output_key=output_key)

    runner.train(model=exp.model, criterion=exp.criterion, optimizer=exp.opt,
                 scheduler=exp.lr_scheduler, loaders=exp.loaders,
                 callbacks=exp.cb_list, **config["runner_params"])
    # Not saving plots if plot_params not specified in config
    if not config.get("plot_params"):
        figs = plot_metrics(logdir=config["runner_params"]["logdir"],
                            metrics=config["plot_params"]["metrics"])
        save_figs(figs, save_dir=config["plot_params"]["save_dir"])

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    with open(args.yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
