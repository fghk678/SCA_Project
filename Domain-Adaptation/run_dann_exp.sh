#!/bin/bash

DATADIR="data"
# 默认值
MODEL="resnet resnet50 2048"
DATA_NAME="Office31"
DATA_FOLDER_NAME="Office31"
SOURCE="A"
TARGET="W"
GPU="0"
EPOCHS=20
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
      echo "用法: $0 [--source 源域] [--target 目标域] [--model 模型类型] [--data_name 数据集名称] [--data_folder_name 数据集文件夹名称] [--gpu GPU_ID] [--epochs 训练轮数] [--seed 种子] [--run 运行次数]"
      echo "    --source: 源域 (默认: A)"
      echo "    --target: 目标域 (默认: W)"
      echo "    --model: 模型类型 (默认: \"resnet resnet50 2048\")"
      echo "    --data_name: 数据集名称 (默认: Office31)"
      echo "    --data_folder_name: 数据集文件夹名称 (默认: office31)"
      echo "    --gpu: 指定GPU ID (默认: 0)"
      echo "    --epochs: 训练轮数 (默认: 20)"
      echo "    --seed: 随机种子 (默认: 33333)"
      echo "    --run: 当前运行次数 (默认: 1)"
      exit 1
      ;;
  esac
done

# 提取模型名称用于目录结构
MODEL_NAME=$(echo $MODEL | cut -d' ' -f2)

# 创建输出目录
OUTPUT_DIR="output/dann/${MODEL_NAME}/${DATA_NAME}"
LOGS_DIR="logs/dann/${MODEL_NAME}/${DATA_NAME}"

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

# JSON文件路径
JSON_FILE="${OUTPUT_DIR}/${SOURCE}_${TARGET}.json"

# 检查JSON文件是否存在，如果不存在则创建
if [ ! -f "$JSON_FILE" ]; then
    echo "{\"results\": []}" > $JSON_FILE
fi

echo "开始运行 $DATA_NAME 数据集上的 DANN 实验: $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\), 模型: $MODEL, GPU: $GPU"

# 创建临时文件用于捕获输出
TEMP_OUTPUT="${LOGS_DIR}/temp_${SOURCE}_${TARGET}_${SEED}.txt"

# 设置GPU并运行实验
LOG_FILE="${LOGS_DIR}/${DATA_NAME}_${SOURCE}_${TARGET}_seed${SEED}.log"

# 运行DANN实验
CUDA_VISIBLE_DEVICES=$GPU python dann.py $DATADIR \
--data_name $DATA_NAME --data_folder_name $DATA_FOLDER_NAME \
--source $SOURCE --target $TARGET --model_type "$MODEL" \
--epochs $EPOCHS --seed $SEED --log_file "$LOG_FILE" 2>&1 | tee $TEMP_OUTPUT

# 提取最后一行包含 "Test accuracy" 的内容
TEST_ACC=$(grep "Test accuracy" $TEMP_OUTPUT | tail -n 1)

# 提取实际的准确率数值
ACC_VALUE=$(echo $TEST_ACC | grep -oP "(?<=Test accuracy: )[0-9.]+")

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
    print(f"原始输出: \"'"$TEST_ACC"'\"")
    sys.exit(1)
'

echo "实验完成，准确率: $ACC_VALUE"
echo "结果已保存到 $JSON_FILE"
echo "日志已保存到 $LOG_FILE"

# 删除临时输出文件
rm $TEMP_OUTPUT

echo "$DATA_NAME 数据集上的 DANN 实验 $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\) 已完成。" 