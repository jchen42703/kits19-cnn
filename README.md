# kits19-cnn
Using convolutional neural networks for the [2019 Kidney and Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/)
## Downloading the Dataset
The recommended way is to just follow the instructions on the [original kits19 Github challenge page](https://github.com/neheller/kits19), which utilizes `git lfs`.
Here is a brief run-down for Google Colaboratory:
```
! sudo add-apt-repository ppa:git-core/ppa
! curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
! sudo apt-get install git-lfs
! git lfs install
% cd "/content/"
! rm -r kits19
! git clone https://github.com/neheller/kits19.git
# takes roughly 11 minutes to download
```
However, this repository also provides an easy way to do it in one line from the command line using a personally uploaded `.tar` file on Google Drive and the `gdown` package. The full code is in `scripts/download.py`. Feel free to just use the `download_kits19` function, especially if you're downloading the dataset from a notebook. <br>
__Note:__ The `.tar` file will be updated regularly based on the new data releases from the actual challenge page.
```
# defaults to have tar extraction
! python kits19-cnn/scripts/download.py --dset_path="/content/" # please specify the dset_path for your specific machine
---------
# for no extraction
python download.py --no-extract --dset_path="/content/"
```
