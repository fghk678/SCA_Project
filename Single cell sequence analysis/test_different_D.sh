#!/bin/bash

# 固定num_anchors=256
anchors=256

# 定义要测试的D值
d_values=(32 64 128 256 512)

# 定义多个随机种子以获得更可靠的结果
seeds=(33333 42 123)

# 基础命令
base_cmd="python train.py --root single_cell/ --data_folder_name processed_data/transcription_factor/ --source rna_seq --target atac_seq --batch_size 32 --workers 4 --orthogonal_w 1.0 --supervised_w 10.0 --encoder_lr 1e-3 --discr_lr 1e-4 --lsmooth 1 --n_epochs 76 --n_z 1 --num_anchors $anchors"

# 创建结果目录
mkdir -p results_D_test

# 为每个D值运行实验
for d_value in "${d_values[@]}"; do
    echo "测试 D=$d_value, num_anchors=$anchors 的结果..."
    
    for seed in "${seeds[@]}"; do
        echo "  - 使用seed=$seed"
        
        # 构建完整命令
        cmd="$base_cmd --D $d_value --seed $seed"
        
        # 运行命令并将输出保存到日志文件
        log_file="results_D_test/D_${d_value}_anchor_${anchors}_seed_${seed}.log"
        echo "运行命令: $cmd"
        echo "输出记录到: $log_file"
        
        $cmd > "$log_file" 2>&1
        
        echo "  - 完成 D=$d_value, seed=$seed"
    done
    
    echo "完成 D=$d_value 的所有实验"
    echo ""
done

echo "所有D值测试完成！" 