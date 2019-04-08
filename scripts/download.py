import gdown
import os
import tarfile
import argparse

def download_kits19(file_id = "1NUZ7pyEActuaZ9sOyGbGUekL3X3ihVjn", extract = True, dset_path = "/content/"):
    """
    Quick script for a quick and easy download that isn't through git lfs.
    This method:
        ETA: 2:21 (143 mb/s) and roughly 4 minutes after that
        Total (w/ extraction): 10 minutes
    Git LFS:
        ETA: ~7-11 minutes (76-80 mb/s)

    Args:
        file_id (str): the unique file id located in a google drive share link (this is provided for this dataset)
        extract (boolean): whether or not you want to extract the downloaded tar file. Defaults to True.
        dset_path (str): Path to the base directory where you want to download and extract the tar file. Defaults to '/content/'
    Returns:
        None
        [The extracted folder will be called: data; feel free to change the name]
    """
    url = 'https://drive.google.com/uc?id={}'.format(file_id)
    f_path = os.path.join(dset_path, 'kits19data.tar')
    gdown.download(url, f_path, quiet=False)
    if extract:
        print('Extracting data [STARTED]')
        tar = tarfile.open(f_path)
        tar.extractall(dset_path)
        print('Extracting data [DONE]')
    return None

def add_bool_arg(parser, name, default=False):
    """
    From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Handles boolean cases from command line through the creating two mutually exclusive arguments: --name and --no-name.

    Args:
        parser (arg.parse.ArgumentParser): the parser you want to add the arguments to
        name: name of the common feature name for the two mutually exclusive arguments; dest = name
        default: default boolean for command line
    Returns:
        None
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='For downloading the KiTS 2019 Dataset')
    add_bool_arg(parser, "extract", default = True) # defaults to extract = True
    parser.add_argument('--dset_path', nargs = 1, type = str, required = False, default = "/content/",
                        help = 'Path to the base directory where you want to download and extract the tar file')
    args = parser.parse_args()
    download_kits19(extract = args.extract, dset_path = args.dset_path[0])
