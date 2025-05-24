#!/bin/bash

DATADIR="data"
# 默认值
MODEL="clip ViT-L/14 768"
DATA_NAME="Office31"
DATA_FOLDER_NAME="Office31"
SOURCE="A"
TARGET="W"
GPU="0"
EPOCHS=10
D=512
SEED=33333
RUN=1
LAMBDAA=1.0
LAMBDAA_CLASSIFY=0.1
LAMBDAA_DIST=1.0
ALPHA=2e-4
BETA=2e-5
CLASS_LR=0.02
TEMP=0
BATCH_SIZE=64
WORKERS=4

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
    --model)
      MODEL="$2"
      shift 2
      ;;
    --data_path)
      DATADIR="$2"
      shift 2
      ;;
    --data_name)
      DATA_NAME="$2"
      shift 2
      ;;
    --data_folder_name)
      DATA_FOLDER_NAME="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --D)
      D="$2"
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
    --lambdaa)
      LAMBDAA="$2"
      shift 2
      ;;
    --lambdaa_classify)
      LAMBDAA_CLASSIFY="$2"
      shift 2
      ;;
    --lambdaa_dist)
      LAMBDAA_DIST="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --class_lr)
      CLASS_LR="$2"
      shift 2
      ;;
    --temp)
      TEMP="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--source 源域] [--target 目标域] [--model 模型类型] [--data_name 数据集名称] [--data_folder_name 数据集文件夹名称] [--gpu GPU_ID] [--epochs 训练轮数] [--seed 种子] [--run 运行次数]"
      exit 1
      ;;
  esac
done

# 提取模型名称用于目录结构
MODEL_NAME=$(echo $MODEL | cut -d' ' -f2)

# 创建输出目录
OUTPUT_DIR="output/proposed_nomcc/${MODEL_NAME}/${DATA_NAME}"
LOGS_DIR="logs/proposed_nomcc/${MODEL_NAME}/${DATA_NAME}"

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

# JSON文件路径
JSON_FILE="${OUTPUT_DIR}/${SOURCE}_${TARGET}.json"

# 检查JSON文件是否存在，如果不存在则创建
if [ ! -f "$JSON_FILE" ]; then
    echo "{\"results\": []}" > $JSON_FILE
fi

echo "开始运行 $DATA_NAME 数据集上的实验: $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\), 模型: $MODEL, GPU: $GPU"

# 创建临时文件用于捕获输出
TEMP_OUTPUT="${LOGS_DIR}/temp_${SOURCE}_${TARGET}_${SEED}.txt"

# 设置GPU并运行实验
LOG_FILE="${LOGS_DIR}/${DATA_NAME}_${SOURCE}_${TARGET}_seed${SEED}.log"

# 运行实验
CUDA_VISIBLE_DEVICES=$GPU python train.py --data_path $DATADIR \
--data_name $DATA_NAME --data_folder_name $DATA_FOLDER_NAME \
--source $SOURCE --target $TARGET --batch_size $BATCH_SIZE --workers $WORKERS \
--lambdaa $LAMBDAA --lambdaa_classify $LAMBDAA_CLASSIFY --lambdaa_dist $LAMBDAA_DIST \
--alpha $ALPHA --beta $BETA --class_lr $CLASS_LR \
--n_epochs $EPOCHS --D $D --seed $SEED --model_type "$MODEL" --temp $TEMP \
--log_file "${LOGS_DIR}/${SOURCE}_${TARGET}_seed${SEED}.log" 2>&1 | tee $TEMP_OUTPUT

# 提取最后一行包含 "The best accuracy for target is" 的内容
BEST_ACC=$(grep "The best accuracy for target is" $TEMP_OUTPUT | tail -n 1)

# 提取实际的准确率数值
ACC_VALUE=$(echo $BEST_ACC | grep -oP "(?<=is )[0-9.]+" | sed 's/\.$//')

# 将结果添加到JSON文件
python3 -c '
import json
import sys
import numpy as np

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
    
    # 计算平均准确率和标准差
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

echo "实验完成，准确率: $ACC_VALUE"
echo "结果已保存到 $JSON_FILE"
echo "日志已保存到 $LOG_FILE"

# 删除临时输出文件
rm $TEMP_OUTPUT

echo "$DATA_NAME 数据集上的实验 $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\) 已完成。" 