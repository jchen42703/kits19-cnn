from glob import glob
from kits19cnn.generators import SliceGenerator
import random

def split_ids(id_list, splits = [0.6, 0.2, 0.2]):
    """
    Divides filenames into train/val/test sets
    Args:
        id_list: list of paths to the case directories
        splits: a list with 3 elements corresponding to the decimal train/val/test splits; [train, val, test]
    Returns:
        a dictionary of file ids for each set
    """
    random.shuffle(id_list)
    total = len(id_list)
    train = round(total * splits[0])
    val_split = round(total * splits[1]) + train
    return {"train": id_list[:train], "val": id_list[train:val_split], "test": id_list[val_split:]
           }

if __name__ == "__main__":
    # getting the filename splits
    base_dir = "/content/kits19/data"
    case_paths = glob(base_dir + "/*/", recursive = True)
    id_dict = split_ids(case_paths)

    batch_size = 5
    n_pos = 0
    input_shape = (512, 512, 1)
    # composed = get_transforms(input_shape)
    # composed = get_only_crop(input_shape, random_crop = True)
    composed = None

    gen = SliceGenerator(temp_ids, batch_size, input_shape = input_shape, n_classes = 2 , n_pos = n_pos, transform = composed)
    gen_val = SliceGenerator(id_dict['val'], batch_size, input_shape = input_shape, n_classes = 2 , n_pos = n_pos, transform = composed)
    ### NOT DONE YET

import argparse
import os
import json

from training_utils import get_model, get_transforms, get_callbacks, get_generators, add_bool_arg

if __name__ == "__main__":
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For training on CNNs on the KiTS19 Challenge Dataset")
    parser.add_argument("--weights_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save your weights")
    parser.add_argument("--dset_path", type = str, required = True,
                        help = "Path to the base kits19/data directory")
    parser.add_argument("--model_name", type = str, required = True,
                        help = "Name of the model you want to train: `cnn`, `capsr3`, `ucapsr3`, or `cnn-simple`")
    parser.add_argument("--epochs", type = int, required = True,
                        help = "Number of epochs")
    parser.add_argument("--batch_size", type = int, required = False, default = 2,
                        help = "Batch size for the CNN should be 17 and for the Capsule Network, it should be 2.")
    parser.add_argument("--n_pos", type = int, required = False, default = 1,
                        help = "Try to make this 1/3 of the batch size (exception for the Capsule Network because its batch size is too small.)")
    parser.add_argument("--lr", type = float, required = False, default = 3e-5,
                        help = "The learning rate")
    parser.add_argument("--steps_per_epoch", type = int, required = False, default = 126,
                        help = "Number of batches per epoch.")
    parser.add_argument("--fold_json_path", type = str, required = False, default = "",
                        help = "Path to the json with the filenames split. If this is not specified, the json will be created in 'weights_dir.'")
    parser.add_argument("--weights_name", type = str, required = False, default = "",
                        help = "Name of the h5 file you want to load the weights from.")
    parser.add_argument("--initial_epoch", type = int, required = False, default = 0,
                        help = "The initial epoch for training.")
    parser.add_argument("--max_queue_size", type = int, required = False, default = 20,
                        help = "Max queue size for training.")
    args = parser.parse_args()
    # Setting up the initial filenames and path
    if args.fold_json_path == "":
        print("Creating the fold...60/20/20 split")
        case_paths = glob(args.dset_path + "/*/", recursive = True)
        id_dict = split_ids(case_paths)
        args.fold_json_path = os.path.join(args.weights_dir, args.model_name + "_fold1.json")
        print("Saving the fold in: ", args.fold_json_path)
        # Saving current fold as a json
        with open(args.fold_json_path, 'w') as fp:
            json.dump(id_dict, fp)
    else:
        with open(args.fold_json_path, 'r') as fp:
            id_dict = json.load(fp)

    # create generators, callbacks, and model
    transform = get_transforms()
    gen, gen_val = get_generators(id_dict, data_dirs, args.batch_size, args.n_pos, transform, steps = args.steps_per_epoch, pos_mask = args.decoder)
    model = get_model(args.model_name, args.lr, decoder = args.decoder)
    callbacks = get_callbacks(os.path.join(args.weights_dir, "checkpoint.h5"), args.decoder)
    # checking if to load weights or not
    weights_path = os.path.join(args.weights_dir, args.weights_name)
    if args.weights_name != "" and os.path.exists(weights_path):
        print("Loading weights from: ", weights_path)
        model.load_weights(weights_path)
    # training
    # feel free to change the settings here if you want to
    print("Starting training...")
    history = model.fit_generator(generator = gen, steps_per_epoch = len(gen), epochs = args.epochs, callbacks = callbacks, validation_data = gen_val,
                                  validation_steps = len(gen_val), max_queue_size = args.max_queue_size, workers = 1, use_multiprocessing = False,
                                  initial_epoch = args.initial_epoch)
    print("Finished training!")
    # save model and history
    history_path = os.path.join(args.weights_dir, args.model_name + "_history.pickle")

    import pickle
    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("Saved the training history in ", history_path)
    weights_path = os.path.join(args.weights_dir, args.model_name + "_weights_" + str(args.epochs) + "epochs.h5")
    model.save(weights_path)
    print("Saved the weights in ", weights_path)
