#!/usr/bin/env bash
DATADIR="/home/lgao638/code/Shared-Component-Analysis/Domain-Adaptation"
# MODEL="resnet resnet50 2048"
MODEL="clip ViT-L/14 768"

python dann.py $DATADIR --data_name Office31 --data_folder_name office31 --source A --target W --model_type "$MODEL" --epochs 1 --seed 1 --log logs/dann/clip/Office31/A2W
