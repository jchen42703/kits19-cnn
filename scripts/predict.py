from catalyst.dl.runner import SupervisedRunner

from kits19cnn.inference import Predictor
from utils import seed_everything
from infer_experiment import SegmentationInferenceExperiment

def main(config):
    """
    Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. experiments/finetune_classification.yml
    Returns:
        None
    """
    # setting up the train/val split with filenames
    seed = config["io_params"]["split_seed"]
    seed_everything(seed)
    mode = config["mode"].lower()
    assert mode in ["classification", "segmentation"], \
        "The `mode` must be one of ['classification', 'segmentation']."
    if mode == "classification":
        raise NotImplementedError
    elif mode == "segmentation":
        exp = SegmentationInferenceExperiment(config)

    print(f"Seed: {seed}\nMode: {mode}")
    pred = Predictor(in_dir=config["in_dir"], out_dir=config["out_dir"],
                     checkpoint_path=config["checkpoint_path"],
                     model=exp.model, test_loader=exp.loaders["test"],
                     pred_3D_params=config["predict_3D_params"])
    pred.run_3D_predictions()

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
