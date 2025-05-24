#!/bin/bash

# DATADIR="/home/lgao638/code/Shared-Component-Analysis/Domain-Adaptation"
DATADIR="data"
MODEL="clip ViT-L/14 768" # 
# MODEL="resnet resnet50 2048"

# 默认值
SOURCE="IN"
TARGET="INR"
SEED=33333
RUN=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --run)
      RUN="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

OUTPUT_DIR="output/proposed/clip_nomcc/imagenet-r"
LOGS_DIR="logs/proposed/clip_nomcc/imagenet-r"
# OUTPUT_DIR="output/proposed/resnet50_nomcc/imagenet-r"
# LOGS_DIR="logs/proposed/resnet50_nomcc/imagenet-r"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

# JSON文件路径
JSON_FILE="${OUTPUT_DIR}/${SOURCE}_${TARGET}.json"

# 检查JSON文件是否存在，如果不存在则创建
if [ ! -f "$JSON_FILE" ]; then
    echo "{\"results\": []}" > $JSON_FILE
fi

echo "开始运行 ImageNet-R 数据集上的实验: $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\)"

# 创建临时文件用于捕获输出
TEMP_OUTPUT="${LOGS_DIR}/temp_${SOURCE}_${TARGET}_${SEED}.txt"

# 运行实验 - 不再需要重定向到temp_output.txt，因为日志系统会处理输出
python train.py --data_path $DATADIR --data_name ImageNetR --data_folder_name ImageNetR \
--source $SOURCE --target $TARGET --batch_size 64 --workers 4 --lambdaa 1.0 --class_layers 1 --seed $SEED \
--lambdaa_classify 0.1 --lambdaa_dist 1.0 --alpha 2e-4 --beta 2e-5 --class_lr 0.02 \
--temp 0 --n_epochs 20 --D 512 --model_type "$MODEL" --n_critic 1 --n_z 1 --class_width 512 \
--log_file "${LOGS_DIR}/${SOURCE}_${TARGET}_seed${SEED}.log" > $TEMP_OUTPUT 2>&1

# 提取最后一行包含 "The best accuracy for target is" 的内容
BEST_ACC=$(grep "The best accuracy for target is" $TEMP_OUTPUT | tail -n 1)

# 提取实际的准确率数值并确保没有多余的小数点
ACC_VALUE=$(echo $BEST_ACC | grep -oP "(?<=is )[0-9.]+" | sed 's/\.$//')

# 将结果添加到JSON文件
python3 -c '
import json
import sys

with open("'"$JSON_FILE"'", "r") as f:
    data = json.load(f)
try:
    acc_value = float("'"$ACC_VALUE"'")
    # 检查是否已有相同run和seed的结果，如果有则更新，没有则添加
    found = False
    for i, result in enumerate(data["results"]):
        if result.get("run") == '"$RUN"' and result.get("seed") == '"$SEED"':
            data["results"][i]["accuracy"] = acc_value
            found = True
            break
    if not found:
        data["results"].append({"run": '"$RUN"', "seed": '"$SEED"', "accuracy": acc_value})
    
    # 计算平均准确率
    import numpy as np
    accuracies = [run["accuracy"] for run in data["results"]]
    data["average_accuracy"] = sum(accuracies) / len(accuracies)
    data["std"] = np.std(accuracies)
    
    with open("'"$JSON_FILE"'", "w") as f:
        json.dump(data, f, indent=4)
except ValueError as e:
    print(f"错误：无法解析准确率 \"{acc_value}\" 为浮点数")
    print(f"原始输出: \"'"$BEST_ACC"'\"")
    sys.exit(1)
'

echo "第 $RUN 次实验完成，准确率: $ACC_VALUE"
echo "结果已保存到 $JSON_FILE"
echo "日志已保存到 ${LOGS_DIR}/${SOURCE}_${TARGET}_seed${SEED}.log"
echo "------------------------------"

# 删除临时输出文件
rm $TEMP_OUTPUT

echo "ImageNet-R 数据集上的实验 $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\) 已完成。" 