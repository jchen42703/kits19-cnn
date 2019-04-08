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
