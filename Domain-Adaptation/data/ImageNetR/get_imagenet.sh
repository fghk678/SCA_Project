#!/bin/bash
#
# script to fully prepare ImageNet dataset
# 1. Download the data
# get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar


# 2. Extract the training data:
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# 3. Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# 4. Download and extract ImageNet-R dataset:
echo "Downloading ImageNet-R dataset..."
if ! wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar; then
    echo "Error: Failed to download imagenet-r.tar"
    echo "Please check your network connection and proxy settings."
    echo "For WSL users, you may need to configure proxy settings."
    exit 1
fi

echo "Download completed successfully!"
echo "Extracting ImageNet-R dataset..."
if tar -xvf imagenet-r.tar; then
    echo "Successfully extracted imagenet-r.tar"
    echo "Cleaning up archive file..."
    rm imagenet-r.tar
    echo "ImageNet-R dataset is ready!"
    echo "Dataset structure:"
    ls -la
else
    echo "Error: Failed to extract imagenet-r.tar"
    exit 1
fi 
