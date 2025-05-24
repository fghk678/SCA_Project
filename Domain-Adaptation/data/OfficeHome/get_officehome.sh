#!/bin/bash

# OfficeHome Dataset Download Script
# This script downloads and extracts all OfficeHome dataset files

echo "Starting OfficeHome dataset download..."

# Files will be extracted to current directory (data/OfficeHome)

# Define download list
declare -a download_list=(
    "image_list:image_list.zip:https://cloud.tsinghua.edu.cn/f/1b0171a188944313b1f5/?dl=1"
    "Art:Art.tgz:https://cloud.tsinghua.edu.cn/f/6a006656b9a14567ade2/?dl=1"
    "Clipart:Clipart.tgz:https://cloud.tsinghua.edu.cn/f/ae88aa31d2d7411dad79/?dl=1"
    "Product:Product.tgz:https://cloud.tsinghua.edu.cn/f/f219b0ff35e142b3ab48/?dl=1"
    "Real_World:Real_World.tgz:https://cloud.tsinghua.edu.cn/f/6c19f3f15bb24ed3951a/?dl=1"
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

echo "OfficeHome dataset download and extraction completed!"
echo "Files are located in: $(pwd)/"

# Show the current directory structure
echo "Dataset structure:"
ls -la

echo "Done!"
