#!/bin/bash
#
# ImageNet-R Dataset Download Script with Multiple Options
# This script provides multiple ways to download ImageNet-R dataset

echo "ImageNet-R Dataset Download Script"
echo "=================================="

# Function to check if file exists and has reasonable size
check_file() {
    if [[ -f "imagenet-r.tar" ]]; then
        local size=$(stat -c%s "imagenet-r.tar" 2>/dev/null || stat -f%z "imagenet-r.tar" 2>/dev/null)
        if [[ $size -gt 1000000 ]]; then  # At least 1MB
            echo "Found existing imagenet-r.tar file (${size} bytes)"
            return 0
        fi
    fi
    return 1
}

# Function to download with wget
download_with_wget() {
    echo "Attempting download with wget..."
    if wget --timeout=30 --tries=3 https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar; then
        return 0
    else
        echo "wget download failed"
        return 1
    fi
}

# Function to download with curl
download_with_curl() {
    echo "Attempting download with curl..."
    if curl -L --connect-timeout 30 --max-time 600 -o imagenet-r.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar; then
        return 0
    else
        echo "curl download failed"
        return 1
    fi
}

# Function to extract the dataset
extract_dataset() {
    echo "Extracting ImageNet-R dataset..."
    if tar -xvf imagenet-r.tar; then
        echo "Successfully extracted imagenet-r.tar"
        echo "Cleaning up archive file..."
        rm imagenet-r.tar
        echo "ImageNet-R dataset is ready!"
        echo "Dataset structure:"
        ls -la
        return 0
    else
        echo "Error: Failed to extract imagenet-r.tar"
        return 1
    fi
}

# Main download logic
if check_file; then
    echo "File already exists, proceeding to extraction..."
else
    echo "Downloading ImageNet-R dataset..."
    
    # Try different download methods
    if ! download_with_wget; then
        echo "Trying alternative download method..."
        if ! download_with_curl; then
            echo ""
            echo "============================================"
            echo "DOWNLOAD FAILED - Manual Download Required"
            echo "============================================"
            echo ""
            echo "Network connection failed. This is common in WSL environments."
            echo ""
            echo "Please try the following solutions:"
            echo ""
            echo "1. Configure WSL proxy (if using proxy):"
            echo "   export http_proxy=http://your-proxy:port"
            echo "   export https_proxy=http://your-proxy:port"
            echo ""
            echo "2. Or manually download the file:"
            echo "   URL: https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
            echo "   Save as: imagenet-r.tar in this directory"
            echo "   Then run this script again"
            echo ""
            echo "3. Alternative download command:"
            echo "   curl -L -o imagenet-r.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
            echo ""
            exit 1
        fi
    fi
    
    echo "Download completed successfully!"
fi

# Extract the dataset
if ! extract_dataset; then
    exit 1
fi

echo "ImageNet-R dataset setup completed!" 