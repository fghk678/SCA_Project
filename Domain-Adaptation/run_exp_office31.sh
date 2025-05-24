#!/bin/bash

DATADIR="/home/lgao638/code/Shared-Component-Analysis/Domain-Adaptation"
# MODEL="clip ViT-L/14 768" # 
MODEL="resnet resnet50 2048"

# 默认值
SOURCE="A"
TARGET="W"

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
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

OUTPUT_DIR="output/proposed/resnet50_noMCC/office31"
LOGS_DIR="logs/proposed/resnet50_noMCC/office31"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

# JSON文件路径
JSON_FILE="${OUTPUT_DIR}/${SOURCE}_${TARGET}.json"
echo "{\"results\": []}" > $JSON_FILE

echo "开始运行 $SOURCE -> $TARGET 的实验..."

# 运行5次实验
for i in {1..5}; do
    echo "运行第 $i 次实验..."
    SEED=$((33333 + $i))
    
    # 创建临时文件用于捕获输出
    TEMP_OUTPUT="${LOGS_DIR}/temp_${SOURCE}_${TARGET}_${SEED}.txt"
    
    # 运行实验 - 使用适合Office31的参数
    python train.py --data_path $DATADIR --data_name Office31 --data_folder_name office31 \
    --source $SOURCE --target $TARGET --batch_size 64 --workers 4 --lambdaa 1.0 --class_layers 1 --seed $SEED \
    --lambdaa_classify 0.1 --lambdaa_dist 1.0 --alpha 2e-4 --beta 2e-5 --class_lr 0.02 \
    --temp 0 --n_epochs 10 --D 256 --model_type "$MODEL" --n_critic 1 --n_z 1 --class_width 512 \
    --log_file "${LOGS_DIR}/office31_${SOURCE}_${TARGET}_seed${SEED}.log" 2>&1 | tee $TEMP_OUTPUT
    
    # 提取最后一行包含 "The best accuracy for target is" 的内容
    BEST_ACC=$(grep "The best accuracy for target is " $TEMP_OUTPUT | tail -n 1)
    
    # 提取实际的准确率数值并确保没有多余的小数点
    ACC_VALUE=$(echo $BEST_ACC | grep -oP "(?<=is )[0-9.]+" | sed 's/\.$//')
    
    # 将结果添加到JSON文件
    python -c "
import json
with open('$JSON_FILE', 'r') as f:
    data = json.load(f)
try:
    acc_value = float('$ACC_VALUE')
    data['results'].append({'run': $i, 'seed': $SEED, 'accuracy': acc_value})
    with open('$JSON_FILE', 'w') as f:
        json.dump(data, f, indent=4)
except ValueError as e:
    print(f'错误：无法解析准确率 \"$ACC_VALUE\" 为浮点数')
    print(f'原始输出: \"$BEST_ACC\"')
    exit(1)
"
    
    echo "第 $i 次实验完成，准确率: $ACC_VALUE"
    echo "结果已保存到 $JSON_FILE"
    echo "日志已保存到 ${LOGS_DIR}/office31_${SOURCE}_${TARGET}_seed${SEED}.log"
    echo "------------------------------"
    
    # 删除临时输出文件
    rm $TEMP_OUTPUT
done

# 计算平均准确率并将其添加到JSON文件
python -c "
import json
with open('$JSON_FILE', 'r') as f:
    data = json.load(f)
accuracies = [run['accuracy'] for run in data['results']]
avg_accuracy = sum(accuracies) / len(accuracies)
data['average_accuracy'] = avg_accuracy
with open('$JSON_FILE', 'w') as f:
    json.dump(data, f, indent=4)
"

echo "所有实验完成。平均准确率已添加到 $JSON_FILE" 