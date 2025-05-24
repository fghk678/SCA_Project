import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 定义anchor值列表和k值列表
anchor_values = [0, 1, 50, 256]
k_values = ['5', '8', '10', '12', '14', '16', '18', '20']  # 不包括'30'以保持与图像一致

# 存储每个anchor值和k值的准确率
anchor_results = defaultdict(lambda: defaultdict(list))

# 读取数据
directory = 'results'
for filename in os.listdir(directory):
    if not filename.endswith('.txt'):
        continue
    
    # 从文件名中提取anchor值
    parts = filename.split('_')
    if len(parts) >= 2 and parts[-2] == 'super':
        anchor_value = int(parts[-1].split('.')[0])
        if anchor_value not in anchor_values:
            continue
        
        # 读取文件的最后一行
        with open(os.path.join(directory, filename), 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip() if lines else None
            
            if not last_line:
                continue
        
        # 解析最后一行数据
        try:
            # 提取JSON部分 - 使用正则表达式找到花括号部分
            match = re.search(r'(\{.*\})', last_line)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                
                # 存储每个k值的结果
                for k in k_values:
                    if k in data:
                        anchor_results[anchor_value][k].append(data[k])
            else:
                print(f"在文件 {filename} 中找不到JSON数据")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

# 计算均值和标准差
anchor_stats = {}
for anchor in anchor_values:
    anchor_stats[anchor] = {
        'mean': {k: np.mean(anchor_results[anchor][k]) for k in k_values if anchor_results[anchor][k]},
        'std': {k: np.std(anchor_results[anchor][k]) for k in k_values if anchor_results[anchor][k]}
    }

# 打印收集到的数据情况
for anchor in anchor_values:
    print(f"anchor={anchor}:")
    for k in k_values:
        if k in anchor_stats[anchor]['mean']:
            print(f"  k={k}: 均值={anchor_stats[anchor]['mean'][k]:.4f}, 标准差={anchor_stats[anchor]['std'][k]:.4f}, 样本数={len(anchor_results[anchor][k])}")
        else:
            print(f"  k={k}: 没有数据")

# 绘制图表，风格接近参考图片
plt.figure(figsize=(10, 7))

# 定义颜色列表 - 使用与参考图片相似的颜色
colors = ['blue', 'orange', 'green', 'red']
line_styles = ['-', '-', '-', '-']
markers = ['o', 'o', 'o', 'o']
labels = [
    'Proposed: No paired', 
    'Proposed: 1 paired', 
    'Proposed: 50 paired', 
    'Proposed: 256 paired'
]

# 将k值转换为数字
x_values = [int(k) for k in k_values]

# 为每个anchor值绘制曲线
for i, anchor in enumerate(anchor_values):
    if not any(k in anchor_stats[anchor]['mean'] for k in k_values):
        print(f"跳过 anchor={anchor}，没有有效数据")
        continue
    
    y_mean = [anchor_stats[anchor]['mean'].get(k, np.nan) for k in k_values]
    y_std = [anchor_stats[anchor]['std'].get(k, 0) for k in k_values]
    
    # 过滤掉nan值
    valid_indices = [i for i, y in enumerate(y_mean) if not np.isnan(y)]
    valid_x = [x_values[i] for i in valid_indices]
    valid_y_mean = [y_mean[i] for i in valid_indices]
    valid_y_std = [y_std[i] for i in valid_indices]
    
    if not valid_x:
        print(f"跳过 anchor={anchor}，没有有效数据点")
        continue
    
    plt.plot(valid_x, valid_y_mean, marker=markers[i], linestyle=line_styles[i], 
             color=colors[i], label=labels[i], linewidth=2, markersize=6)
    plt.fill_between(valid_x, 
                     [y - s for y, s in zip(valid_y_mean, valid_y_std)],
                     [y + s for y, s in zip(valid_y_mean, valid_y_std)],
                     color=colors[i], alpha=0.2)

# 设置坐标轴范围
plt.ylim(0, 0.80)
plt.xlim(4, 21)

# 设置图表属性
plt.xlabel('K of the KNN evaluation metric', fontsize=22)
plt.ylabel('KNN accuracy', fontsize=22)
plt.legend(loc='upper left', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

# 保存图表
plt.savefig('anchor_comparison_plot.png', dpi=300, bbox_inches='tight')
print("图表已保存为 anchor_comparison_plot.png") 