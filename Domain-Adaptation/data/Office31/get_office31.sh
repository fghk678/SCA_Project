#!/bin/bash

# Office31 Dataset Download Script
# This script downloads and extracts all Office31 dataset files

echo "Starting Office31 dataset download..."

# Files will be extracted to current directory (data/Office31)

# Define download list
declare -a download_list=(
    "image_list:image_list.zip:https://cloud.tsinghua.edu.cn/f/2c1dd9fbcaa9455aa4ad/?dl=1"
    "amazon:amazon.tgz:https://cloud.tsinghua.edu.cn/f/ec12dfcddade43ab8101/?dl=1"
    "dslr:dslr.tgz:https://cloud.tsinghua.edu.cn/f/a41d818ae2f34da7bb32/?dl=1"
    "webcam:webcam.tgz:https://cloud.tsinghua.edu.cn/f/8a41009a166e4131adcd/?dl=1"
)

# Function to download and extract files
download_and_extract() {
    local name=$1
    local filename=$2
    local url=$3
    
    echo "Downloading $name ($filename)..."
    
    # Download file
    if ! wget -O "$filename" "$url"; then
        echo "Error: Failed to download $filename"
        return 1
    fi
    
    echo "Downloaded $filename successfully"
    
    # Extract based on file extension
    if [[ $filename == *.zip ]]; then
        echo "Extracting $filename..."
        if ! unzip -q "$filename"; then
            echo "Error: Failed to extract $filename"
            return 1
        fi
    elif [[ $filename == *.tgz ]] || [[ $filename == *.tar.gz ]]; then
        echo "Extracting $filename..."
        if ! tar -xzf "$filename"; then
            echo "Error: Failed to extract $filename"
            return 1
        fi
    fi
    
    echo "Extracted $filename successfully"
    
    # Remove downloaded archive to save space
    rm "$filename"
    echo "Cleaned up $filename"
    
    return 0
}

# Download and extract each file
for item in "${download_list[@]}"; do
    IFS=':' read -r name filename url <<< "$item"
    
    if ! download_and_extract "$name" "$filename" "$url"; then
        echo "Failed to process $name. Continuing with next file..."
        continue
    fi
    
    echo "Successfully processed $name"
    echo "----------------------------------------"
done

echo "Office31 dataset download and extraction completed!"
echo "Files are located in: $(pwd)/"

# Show the current directory structure
echo "Dataset structure:"
ls -la

echo "Done!"
