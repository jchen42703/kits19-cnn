import argparse
from kits19cnn.io.preprocess import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For downloading and preprocessing the dataset.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help = "Path to the base data directory: kits19/data")
    parser.add_argument("--output_path", type=str, required=True,
                        help = "Path to the base directory where you want to save your preprocessed dataset")
    parser.add_argument("--clip_low", type=float, required=False, default=-75.75658734213053,
                        help="Lower-bound to clip to. Default is the 0.05 percentile of the dataset's ROI pixels.")
    parser.add_argument("--clip_upper", type=float, required=False, default=349.4891265535317,
                        help="Upper-bound to clip to. Default is the 99.5 percentile of the dataset's ROI pixels.")
    args = parser.parse_args()

    clip = [args.clip_low, args.clip_upper]
    print("Input Directory: {0}\nOutput Directory: {1}".format(args.dset_path, args.output_path))
    preprocess = Preprocessor(args.dset_path, args.output_path, clip_values=clip)
    preprocess.gen_data()
