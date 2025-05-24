import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置中文字体显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 定义D值列表和k值列表
d_values = [32, 64, 128, 256, 512]
k_values = ['5', '8', '10', '12', '14', '16', '18', '20']  # 不包括'30'以保持与图像一致

# 存储每个D值和k值的准确率
d_results = defaultdict(lambda: defaultdict(list))

# 读取数据
directory = 'results_D_test/D-variation'
for filename in os.listdir(directory):
    if not filename.endswith('.txt'):
        continue
    
    # 从文件名中提取D值
    parts = filename.split('_')
    d_value = int(parts[-3])
    
    # 读取文件的最后一行
    with open(os.path.join(directory, filename), 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
    
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
                    d_results[d_value][k].append(data[k])
        else:
            print(f"在文件 {filename} 中找不到JSON数据")
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")

# 计算均值和标准差
d_stats = {}
for d in d_values:
    d_stats[d] = {
        'mean': {k: np.mean(d_results[d][k]) for k in k_values if d_results[d][k]},
        'std': {k: np.std(d_results[d][k]) for k in k_values if d_results[d][k]}
    }

# 打印收集到的数据情况
for d in d_values:
    print(f"D={d}:")
    for k in k_values:
        if k in d_stats[d]['mean']:
            print(f"  k={k}: 均值={d_stats[d]['mean'][k]:.4f}, 标准差={d_stats[d]['std'][k]:.4f}, 样本数={len(d_results[d][k])}")
        else:
            print(f"  k={k}: 没有数据")

# 绘制图表，风格接近参考图片
plt.figure(figsize=(10, 7))

# 定义颜色列表 - 使用与参考图片相似的颜色
colors = ['blue', 'orange', 'green', 'red', 'purple']
line_styles = ['-', '-', '-', '-', '-']
markers = ['o', 'o', 'o', 'o', 'o']
labels = ['D=32', 'D=64', 'D=128', 'D=256', 'D=512']

# 将k值转换为数字
x_values = [int(k) for k in k_values]

# 为每个D值绘制曲线
for i, d in enumerate(d_values):
    if not any(k in d_stats[d]['mean'] for k in k_values):
        print(f"跳过 D={d}，没有有效数据")
        continue
    
    y_mean = [d_stats[d]['mean'].get(k, np.nan) for k in k_values]
    y_std = [d_stats[d]['std'].get(k, 0) for k in k_values]
    
    # 过滤掉nan值
    valid_indices = [i for i, y in enumerate(y_mean) if not np.isnan(y)]
    valid_x = [x_values[i] for i in valid_indices]
    valid_y_mean = [y_mean[i] for i in valid_indices]
    valid_y_std = [y_std[i] for i in valid_indices]
    
    if not valid_x:
        print(f"跳过 D={d}，没有有效数据点")
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
# plt.title('KNN准确率比较 (num_anchors=256)', fontsize=14)
plt.legend(loc='upper left', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# 保存图表
plt.savefig('D_comparison_plot.png', dpi=300, bbox_inches='tight')
print("图表已保存为 D_comparison_plot.png") 