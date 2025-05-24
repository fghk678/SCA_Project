#!/bin/bash
# bash run_train_all.sh --gpus 0,1
# bash run_train_all.sh --model "clip ViT-L/14 768" --data_name OfficeHome --data_folder_name OfficeHome
# 默认配置
GPUS=(0 1)
CONCURRENT_PER_GPU=2
MODEL="clip ViT-L/14 768"
DATA_NAME="Office31"
DATA_FOLDER_NAME="Office31"
EPOCHS=10
D=512
NUM_RUNS=5  # 每个源域-目标域组合的实验运行次数
LAMBDAA=1.0
LAMBDAA_CLASSIFY=0.1
LAMBDAA_DIST=1.0
ALPHA=2e-4
BETA=2e-5
CLASS_LR=0.02
TEMP=0.55
BATCH_SIZE=64
WORKERS=4

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
    --model)
      MODEL="$2"
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
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --D)
      D="$2"
      shift 2
      ;;
    --runs)
      NUM_RUNS=$2
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
      echo "用法: $0 [--gpus gpu_ids] [--concurrent 并行数] [--model 模型类型] [--data_name 数据集名称] [--data_folder_name 数据集文件夹名称] [--epochs 训练轮数] [--runs 实验次数]"
      echo "    --gpus: 逗号分隔的GPU ID列表，例如 '0,1' (默认: '0,1')"
      echo "    --concurrent: 每个GPU上并行运行的实验数量 (默认: 2)"
      echo "    --model: 模型类型 (默认: \"clip ViT-L/14 768\")"
      echo "    --data_name: 数据集名称 (默认: Office31)"
      echo "    --data_folder_name: 数据集文件夹名称 (默认: office31)"
      echo "    --epochs: 训练轮数 (默认: 10)"
      echo "    --D: 共享组件数 (默认: 512)"
      echo "    --runs: 每个源域-目标域组合的实验运行次数 (默认: 5)"
      exit 1
      ;;
  esac
done

# 提取模型名称用于目录结构
MODEL_NAME=$(echo $MODEL | cut -d' ' -f2)

OUTPUT_DIR="output/proposed/${MODEL_NAME}/${DATA_NAME}"
mkdir -p $OUTPUT_DIR

# 显示配置信息
GPU_COUNT=${#GPUS[@]}
echo "使用 ${GPU_COUNT} 个GPU: ${GPUS[*]}"
echo "每个GPU并行运行 ${CONCURRENT_PER_GPU} 个实验"
echo "模型: $MODEL"
echo "数据集: $DATA_NAME ($DATA_FOLDER_NAME)"
echo "训练轮数: $EPOCHS"
echo "共享组件数: $D"
echo "每个源域-目标域组合运行 ${NUM_RUNS} 次实验"

# 设置域列表
if [ "$DATA_NAME" == "Office31" ]; then
    DOMAINS=("A" "D" "W")
elif [ "$DATA_NAME" == "OfficeHome" ]; then
    DOMAINS=("Ar" "Cl" "Pr" "Rw")
elif [ "$DATA_NAME" == "VisDA" ]; then
    DOMAINS=("S" "T")
elif [ "$DATA_NAME" == "ImageNetR" ]; then
    DOMAINS=("IN" "IN-val" "INR")
    DATADIR="/data/gaolufei/datasets/ttadata"
else
    echo "未知数据集: $DATA_NAME"
    exit 1
fi

# 创建任务列表
declare -a TASKS=()
for SOURCE in "${DOMAINS[@]}"; do
    if [ "$SOURCE" == "INR" ]; then
        continue
    fi
    for TARGET in "${DOMAINS[@]}"; do
        if [ "$TARGET" == "IN" -o "$TARGET" == "IN-val" ]; then
            continue
        fi
        # 跳过源域和目标域相同的情况
        if [ "$SOURCE" != "$TARGET" ]; then
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
            
            echo "已有的 $SOURCE -> $TARGET 实验种子: $EXISTING_SEEDS"
            
            for RUN in $(seq 1 $NUM_RUNS); do
                SEED=$((33333 + $RUN))
                
                # 检查该种子的实验是否已完成
                if [[ "$EXISTING_SEEDS" == *"$SEED"* ]]; then
                    echo "跳过已完成的实验: $SOURCE -> $TARGET (种子: $SEED)"
                else
                    TASKS+=("$SOURCE,$TARGET,$SEED,$RUN")
                    echo "添加任务: $SOURCE -> $TARGET (种子: $SEED)"
                fi
            done
        fi
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "总共有 $TOTAL_TASKS 个实验任务需要运行"

# 如果没有任务需要运行，则退出
if [ $TOTAL_TASKS -eq 0 ]; then
    echo "没有需要运行的实验任务，所有实验已完成。"
    
    # 计算所有实验的平均结果
    echo "计算总体平均结果..."
    
    python3 -c '
import json
import os
import glob
import numpy as np

output_dir = "'"$OUTPUT_DIR"'"
all_files = glob.glob(f"{output_dir}/*.json")
all_accuracies = []
domain_results = {}

for file_path in all_files:
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "average_accuracy" in data:
        domains = os.path.basename(file_path).replace(".json", "")
        source, target = domains.split("_")
        all_accuracies.append(data["average_accuracy"])
        domain_results[f"{source}->{target}"] = data["average_accuracy"]

# 计算总平均准确率
if all_accuracies:
    overall_avg = sum(all_accuracies) / len(all_accuracies)
    overall_std = np.std(all_accuracies)
    
    # 将总体结果保存到文件
    with open(f"{output_dir}/overall_results.json", "w") as f:
        json.dump({
            "model": "'"$MODEL"'",
            "data_name": "'"$DATA_NAME"'",
            "domain_results": domain_results,
            "overall_average_accuracy": overall_avg,
            "overall_std": overall_std
        }, f, indent=4)
    
    print(f"总体平均准确率: {overall_avg:.4f} ± {overall_std:.4f}")
    print(f"详细结果已保存到 {output_dir}/overall_results.json")
else:
    print("没有找到有效的实验结果")
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
                IFS=',' read -r SOURCE TARGET SEED RUN <<< "$TASK"
                
                echo "====================================="
                echo "在GPU $GPU 上运行实验 $((CURRENT_TASK_INDEX+1))/$TOTAL_TASKS: $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\)"
                echo "====================================="
                
                # 后台运行实验，指定GPU
                (
                    bash run_train_exp.sh --source "$SOURCE" --target "$TARGET" --data_path "$DATADIR" \
                        --model "$MODEL" --data_name "$DATA_NAME" --data_folder_name "$DATA_FOLDER_NAME" \
                        --gpu "$GPU" --epochs "$EPOCHS" --D "$D" \
                        --lambdaa "$LAMBDAA" --lambdaa_classify "$LAMBDAA_CLASSIFY" --lambdaa_dist "$LAMBDAA_DIST" \
                        --alpha "$ALPHA" --beta "$BETA" --class_lr "$CLASS_LR" --temp "$TEMP" \
                        --batch_size "$BATCH_SIZE" --workers "$WORKERS" \
                        --seed "$SEED" --run "$RUN"
                        
                    echo "====================================="
                    echo "GPU $GPU 上的实验 $SOURCE -> $TARGET \(第 $RUN 次实验, 种子: $SEED\) 已完成"
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

# 计算所有实验的平均结果
echo "所有域适应实验已完成，计算总体平均结果..."

python3 -c '
import json
import os
import glob
import numpy as np

output_dir = "'"$OUTPUT_DIR"'"
all_files = glob.glob(f"{output_dir}/*.json")
all_accuracies = []
domain_results = {}

for file_path in all_files:
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "average_accuracy" in data:
        domains = os.path.basename(file_path).replace(".json", "")
        source, target = domains.split("_")
        all_accuracies.append(data["average_accuracy"])
        domain_results[f"{source}->{target}"] = data["average_accuracy"]

# 计算总平均准确率
if all_accuracies:
    overall_avg = sum(all_accuracies) / len(all_accuracies)
    overall_std = np.std(all_accuracies)
    
    # 将总体结果保存到文件
    with open(f"{output_dir}/overall_results.json", "w") as f:
        json.dump({
            "model": "'"$MODEL"'",
            "data_name": "'"$DATA_NAME"'",
            "domain_results": domain_results,
            "overall_average_accuracy": overall_avg,
            "overall_std": overall_std
        }, f, indent=4)
    
    print(f"总体平均准确率: {overall_avg:.4f} ± {overall_std:.4f}")
    print(f"详细结果已保存到 {output_dir}/overall_results.json")
else:
    print("没有找到有效的实验结果")
'

echo "所有实验已完成!" 