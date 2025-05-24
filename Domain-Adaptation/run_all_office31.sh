#!/bin/bash

# 默认配置
GPUS=(0 1)
CONCURRENT_PER_GPU=2

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
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--gpus gpu_ids] [--concurrent 并行数]"
      echo "    --gpus: 逗号分隔的GPU ID列表，例如 '0,1'"
      echo "    --concurrent: 每个GPU上并行运行的实验数量 (默认: 2)"
      exit 1
      ;;
  esac
done

# 显示配置信息
GPU_COUNT=${#GPUS[@]}
echo "使用 ${GPU_COUNT} 个GPU: ${GPUS[*]}"
echo "每个GPU并行运行 ${CONCURRENT_PER_GPU} 个实验"

# Office31的所有域
DOMAINS=("A" "D" "W")

# 创建任务列表
declare -a TASKS=()
for SOURCE in "${DOMAINS[@]}"; do
    for TARGET in "${DOMAINS[@]}"; do
        # 跳过源域和目标域相同的情况
        if [ "$SOURCE" != "$TARGET" ]; then
            TASKS+=("$SOURCE,$TARGET")
        fi
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "总共有 $TOTAL_TASKS 个实验任务需要运行"

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
                IFS=',' read -r SOURCE TARGET <<< "$TASK"
                
                echo "====================================="
                echo "在GPU $GPU 上运行实验 $((CURRENT_TASK_INDEX+1))/$TOTAL_TASKS: $SOURCE -> $TARGET"
                echo "====================================="
                
                # 后台运行实验，设置CUDA_VISIBLE_DEVICES
                (
                    CUDA_VISIBLE_DEVICES=$GPU bash run_exp_office31.sh --source "$SOURCE" --target "$TARGET"
                    echo "====================================="
                    echo "GPU $GPU 上的实验 $SOURCE -> $TARGET 已完成"
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

python -c "
import json
import os
import glob

output_dir = 'output/proposed/office31'
all_files = glob.glob(f'{output_dir}/*.json')
all_accuracies = []
domain_results = {}

for file_path in all_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'average_accuracy' in data:
        domains = os.path.basename(file_path).replace('.json', '')
        source, target = domains.split('_')
        all_accuracies.append(data['average_accuracy'])
        domain_results[f'{source}->{target}'] = data['average_accuracy']

# 计算总平均准确率
if all_accuracies:
    overall_avg = sum(all_accuracies) / len(all_accuracies)
    
    # 将总体结果保存到文件
    with open(f'{output_dir}/overall_results.json', 'w') as f:
        json.dump({
            'domain_results': domain_results,
            'overall_average_accuracy': overall_avg
        }, f, indent=4)
    
    print(f'总体平均准确率: {overall_avg:.4f}')
    print(f'详细结果已保存到 {output_dir}/overall_results.json')
else:
    print('没有找到有效的实验结果')
"

echo "所有实验已完成!" 