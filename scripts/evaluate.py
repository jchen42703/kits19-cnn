from kits19cnn.inference import Evaluator

def main(config):
    """
    Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. script_configs/eval.yml
    Returns:
        None
    """
    evaluator = Evaluator(config["orig_img_dir"], config["pred_dir"],
                          label_file_ending=config["label_file_ending"])
    evaluator.evaluate_all(print_metrics=config["print_metrics"])

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
