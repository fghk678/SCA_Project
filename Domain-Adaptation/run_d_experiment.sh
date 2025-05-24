#!/bin/bash

# 默认配置
GPUS=(0 1)
CONCURRENT_PER_GPU=1
NUM_RUNS=1  # 每个D值的实验运行次数
D_VALUES=(64 128 256 512 1024)  # 需要测试的D值
SOURCE="IN-val"  # 默认源域
TARGET="INR"  # 默认目标域

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      # 将逗号分隔的GPU列表转换为数组
      IFS=',' read -ra GPUS <<< "$2"
      shift 2
      ;;
    --concurrent)
      CONCURRENT_PER_GPU=$2
      shift 2
      ;;
    --runs)
      NUM_RUNS=$2
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --d_values)
      # 将逗号分隔的D值列表转换为数组
      IFS=',' read -ra D_VALUES <<< "$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--gpus gpu_ids] [--concurrent 并行数] [--runs 实验次数] [--source 源域] [--target 目标域] [--d_values D值列表]"
      echo "    --gpus: 逗号分隔的GPU ID列表，例如 '0,1'"
      echo "    --concurrent: 每个GPU上并行运行的实验数量 (默认: 1)"
      echo "    --runs: 每个D值的实验运行次数 (默认: 5)"
      echo "    --source: 源域 (默认: IN)"
      echo "    --target: 目标域 (默认: INR)"
      echo "    --d_values: 逗号分隔的D值列表，例如 '64,128,256,512'"
      exit 1
      ;;
  esac
done

# 显示配置信息
GPU_COUNT=${#GPUS[@]}
echo "使用 ${GPU_COUNT} 个GPU: ${GPUS[*]}"
echo "每个GPU并行运行 ${CONCURRENT_PER_GPU} 个实验"
echo "每个D值运行 ${NUM_RUNS} 次实验"
echo "测试的D值: ${D_VALUES[*]}"
echo "源域: $SOURCE, 目标域: $TARGET"

# 基础输出目录
# BASE_OUTPUT_DIR="output/proposed/clip_nomcc/imagenet-r"
# BASE_LOGS_DIR="logs/proposed/clip_nomcc/imagenet-r"
BASE_OUTPUT_DIR="output/proposed/resnet50_nomcc/imagenet-r"
BASE_LOGS_DIR="logs/proposed/resnet50_nomcc/imagenet-r"

# 创建任务列表
declare -a TASKS=()

for D_VALUE in "${D_VALUES[@]}"; do
    # 为每个D值创建特定的输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/D${D_VALUE}"
    mkdir -p $OUTPUT_DIR
    
    # 检查JSON文件是否存在
    JSON_FILE="${OUTPUT_DIR}/${SOURCE}_${TARGET}.json"
    if [ -f "$JSON_FILE" ]; then
        # 读取现有结果
        EXISTING_SEEDS=$(python3 -c '
import json
import sys
try:
    with open("'"$JSON_FILE"'", "r") as f:
        data = json.load(f)
    seeds = [result.get("seed") for result in data.get("results", [])]
    print(",".join(map(str, seeds)))
except Exception as e:
    print("")
')
    else
        EXISTING_SEEDS=""
    fi
    
    echo "已有的 D=$D_VALUE, $SOURCE -> $TARGET 实验种子: $EXISTING_SEEDS"
    
    for RUN in $(seq 1 $NUM_RUNS); do
        SEED=$((33333 + $RUN))
        
        # 检查该种子的实验是否已完成
        if [[ "$EXISTING_SEEDS" == *"$SEED"* ]]; then
            echo "跳过已完成的实验: D=$D_VALUE, $SOURCE -> $TARGET \(种子: $SEED\)"
        else
            TASKS+=("$D_VALUE,$SOURCE,$TARGET,$SEED,$RUN")
            echo "添加任务: D=$D_VALUE, $SOURCE -> $TARGET \(种子: $SEED\)"
        fi
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "总共有 $TOTAL_TASKS 个实验任务需要运行"

# 如果没有任务需要运行，则退出
if [ $TOTAL_TASKS -eq 0 ]; then
    echo "没有需要运行的实验任务，所有实验已完成。"
    
    # 计算所有D值的平均结果并比较
    echo "计算并比较不同D值的结果..."
    
    python3 -c '
import json
import os
import glob
import numpy as np

base_output_dir = "'"$BASE_OUTPUT_DIR"'"
d_values = ['"${D_VALUES[@]}"']
d_values = [int(d) for d in d_values]
source = "'"$SOURCE"'"
target = "'"$TARGET"'"
d_results = {}

for d_value in d_values:
    output_dir = f"{base_output_dir}/D{d_value}"
    json_file = f"{output_dir}/{source}_{target}.json"
    
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if "average_accuracy" in data:
            d_results[d_value] = {
                "average_accuracy": data["average_accuracy"],
                "std": data.get("std", 0)
            }

# 打印结果比较
print(f"\n源域: {source}, 目标域: {target}的不同D值结果比较:")
print(f"{'D值':<10}{'平均准确率':<15}{'标准差':<10}")
print("-" * 35)

for d_value in sorted(d_results.keys()):
    result = d_results[d_value]
    print(f"{d_value:<10}{result['average_accuracy']:<15.4f}{result['std']:<10.4f}")

# 将比较结果保存到文件
comparison_file = f"{base_output_dir}/d_comparison_{source}_{target}.json"
with open(comparison_file, "w") as f:
    json.dump({
        "source": source,
        "target": target,
        "d_results": {str(k): v for k, v in d_results.items()}
    }, f, indent=4)

print(f"\n比较结果已保存到 {comparison_file}")
'
    
    exit 0
fi

# 计算每个批次的实验数量
BATCH_SIZE=$((GPU_COUNT * CONCURRENT_PER_GPU))

# 运行所有任务
TASK_INDEX=0
while [ $TASK_INDEX -lt $TOTAL_TASKS ]; do
    PIDS=()
    
    # 为每个GPU分配任务
    for ((gpu_idx=0; gpu_idx<GPU_COUNT; gpu_idx++)); do
        GPU=${GPUS[$gpu_idx]}
        
        # 在当前GPU上运行CONCURRENT_PER_GPU个任务
        for ((concurrent=0; concurrent<CONCURRENT_PER_GPU; concurrent++)); do
            CURRENT_TASK_INDEX=$((TASK_INDEX + gpu_idx * CONCURRENT_PER_GPU + concurrent))
            
            # 检查是否还有任务
            if [ $CURRENT_TASK_INDEX -lt $TOTAL_TASKS ]; then
                TASK=${TASKS[$CURRENT_TASK_INDEX]}
                IFS=',' read -r D_VALUE SOURCE TARGET SEED RUN <<< "$TASK"
                
                # 为当前D值创建输出目录
                OUTPUT_DIR="${BASE_OUTPUT_DIR}/D${D_VALUE}"
                LOGS_DIR="${BASE_LOGS_DIR}/D${D_VALUE}"
                mkdir -p $OUTPUT_DIR
                mkdir -p $LOGS_DIR
                
                echo "====================================="
                echo "在GPU $GPU 上运行实验 $((CURRENT_TASK_INDEX+1))/$TOTAL_TASKS: D=$D_VALUE, $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\)"
                echo "====================================="
                
                # 后台运行实验，设置CUDA_VISIBLE_DEVICES
                (
                    # 创建临时文件用于捕获输出
                    TEMP_OUTPUT="${LOGS_DIR}/temp_${SOURCE}_${TARGET}_D${D_VALUE}_${SEED}.txt"
                    LOG_FILE="${LOGS_DIR}/${SOURCE}_${TARGET}_D${D_VALUE}_seed${SEED}.log"
                    
                    # JSON文件路径
                    JSON_FILE="${OUTPUT_DIR}/${SOURCE}_${TARGET}.json"
                    
                    # 检查JSON文件是否存在，如果不存在则创建
                    if [ ! -f "$JSON_FILE" ]; then
                        echo "{\"results\": []}" > $JSON_FILE
                    fi
                    
                    # 运行实验
                    CUDA_VISIBLE_DEVICES=$GPU python train.py --data_path /data/gaolufei/datasets/ttadata \
                    --data_name ImageNetR --data_folder_name ImageNetR \
                    --source $SOURCE --target $TARGET --batch_size 64 --workers 4 --lambdaa 1.0 --class_layers 1 --seed $SEED \
                    --lambdaa_classify 0.1 --lambdaa_dist 1.0 --alpha 2e-4 --beta 2e-5 --class_lr 0.02 \
                    --temp 0 --n_epochs 20 --D $D_VALUE --model_type "resnet resnet50 2048" --n_critic 1 --n_z 1 --class_width 512 \
                    --log_file "$LOG_FILE" > $TEMP_OUTPUT 2>&1
                    
                    # 提取最后一行包含 "The best accuracy for target is" 的内容
                    BEST_ACC=$(grep "The best accuracy for target is" $TEMP_OUTPUT | tail -n 1)
                    
                    # 提取实际的准确率数值并确保没有多余的小数点
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
        data["results"].append({"run": '"$RUN"', "seed": '"$SEED"', "accuracy": acc_value, "d_value": '"$D_VALUE"'})
    
    # 计算平均准确率
    accuracies = [run["accuracy"] for run in data["results"]]
    data["average_accuracy"] = sum(accuracies) / len(accuracies)
    data["std"] = np.std(accuracies)
    data["d_value"] = '"$D_VALUE"'
    
    with open("'"$JSON_FILE"'", "w") as f:
        json.dump(data, f, indent=4)
except ValueError as e:
    print(f"错误：无法解析准确率 \"{acc_value}\" 为浮点数")
    print(f"原始输出: \"'"$BEST_ACC"'\"")
    sys.exit(1)
'
                    
                    echo "实验完成，准确率: $ACC_VALUE"
                    echo "结果已保存到 $JSON_FILE"
                    
                    # 删除临时输出文件
                    rm $TEMP_OUTPUT
                    
                    echo "====================================="
                    echo "GPU $GPU 上的实验 D=$D_VALUE, $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\) 已完成"
                    echo "====================================="
                ) &
                
                # 保存进程ID
                PIDS+=($!)
                
                # 稍微延迟以避免同时启动造成的资源竞争
                sleep 2
            fi
        done
    done
    
    # 等待当前批次的所有任务完成
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    # 更新任务索引
    TASK_INDEX=$((TASK_INDEX + BATCH_SIZE))
    
    echo "当前批次实验完成"
done

# 计算所有D值的平均结果并比较
echo "所有域适应实验已完成，计算并比较不同D值的结果..."

python3 -c '
import json
import os
import glob
import numpy as np

base_output_dir = "'"$BASE_OUTPUT_DIR"'"
d_values = ['"${D_VALUES[@]}"']
d_values = [int(d) for d in d_values]
source = "'"$SOURCE"'"
target = "'"$TARGET"'"
d_results = {}

for d_value in d_values:
    output_dir = f"{base_output_dir}/D{d_value}"
    json_file = f"{output_dir}/{source}_{target}.json"
    
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if "average_accuracy" in data:
            d_results[d_value] = {
                "average_accuracy": data["average_accuracy"],
                "std": data.get("std", 0)
            }

# 打印结果比较
print(f"\n源域: {source}, 目标域: {target}的不同D值结果比较:")
print(f"{'D值':<10}{'平均准确率':<15}{'标准差':<10}")
print("-" * 35)

for d_value in sorted(d_results.keys()):
    result = d_results[d_value]
    print(f"{d_value:<10}{result['average_accuracy']:<15.4f}{result['std']:<10.4f}")

# 将比较结果保存到文件
comparison_file = f"{base_output_dir}/d_comparison_{source}_{target}.json"
with open(comparison_file, "w") as f:
    json.dump({
        "source": source,
        "target": target,
        "d_results": {str(k): v for k, v in d_results.items()}
    }, f, indent=4)

print(f"\n比较结果已保存到 {comparison_file}")
'

echo "所有D值实验已完成!" 