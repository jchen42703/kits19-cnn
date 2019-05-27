from glob import glob
from kits19cnn.generators import SliceGenerator

def split_ids(id_list, splits = [0.6, 0.2, 0.2]):
    """
    Divides filenames into train/val/test sets
    Args:
        id_list: list of paths to the case directories
        splits: a list with 3 elements corresponding to the decimal train/val/test splits; [train, val, test]
    Returns:
        a dictionary of file ids for each set
    """
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

    temp_ids = ["/content/kits19/data/case_00160/" for i in range(100)]
    gen = SliceGenerator(temp_ids, batch_size, input_shape = input_shape, n_classes = 2 , n_pos = n_pos, transform = composed)
    gen_val = SliceGenerator(id_dict['val'], batch_size, input_shape = input_shape, n_classes = 2 , n_pos = n_pos, transform = composed)
    ### NOT DONE YET
