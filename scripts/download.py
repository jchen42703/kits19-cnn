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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='For downloading the KiTS 2019 Dataset')
    parser.add_argument('--extract', nargs = 1, type = bool, required = False, default = True,
                        help = 'Boolean on whether or not you want to extract the downloaded tar')
    parser.add_argument('--dset_path', nargs = 1, type = str, required = False, default = "/content/",
                        help = 'Path to the base directory where you want to download and extract the tar file')
    args = parser.parse_args()
    download_kits19(extract = args.extract, dset_path = args.dset_path)
