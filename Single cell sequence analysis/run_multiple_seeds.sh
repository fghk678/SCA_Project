#!/bin/bash

# 定义要测试的num_anchors值
anchor_values=(256 50 1 0)
# 定义10个不同的随机种子
seeds=(33333 42 123 456 789 1024 2048 3072 4096 5120)

# 基础命令
base_cmd="python train.py --root single_cell/ --data_folder_name processed_data/transcription_factor/ --source rna_seq --target atac_seq --batch_size 32 --workers 4 --orthogonal_w 1.0 --supervised_w 10.0 --encoder_lr 1e-3 --discr_lr 1e-4 --lsmooth 1 --n_epochs 76 --D 256 --n_z 1"

# 创建结果目录
mkdir -p results

# 为每个anchor值运行10次实验
for anchor in "${anchor_values[@]}"; do
    echo "运行 num_anchors=$anchor 的实验..."
    
    for seed in "${seeds[@]}"; do
        echo "  - 使用seed=$seed"
        
        # 构建完整命令
        cmd="$base_cmd --num_anchors $anchor --seed $seed"
        
        # 运行命令并将输出保存到日志文件
        log_file="results/anchor_${anchor}_seed_${seed}.log"
        echo "运行命令: $cmd"
        echo "输出记录到: $log_file"
        
        $cmd > "$log_file" 2>&1
        
        echo "  - 完成 seed=$seed"
    done
    
    echo "完成 num_anchors=$anchor 的所有实验"
    echo ""
done

echo "所有实验完成！" 