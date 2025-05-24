import os
import json
import glob
import re

def extract_accuracy_from_log(log_file):
    """从日志文件中提取最后一行的准确率"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # 获取最后一行
            last_line = lines[-1].strip()
            # 使用正则表达式提取准确率
            match = re.search(r'test_acc1 = (\d+\.\d+)', last_line)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"处理文件 {log_file} 时出错: {str(e)}")
    return None

def update_json_with_accuracy(log_dir, output_dir):
    """更新JSON文件中的准确率"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(log_dir, "**/*.txt"), recursive=True)
    
    for txt_file in txt_files:
        # 从文件路径中提取源域和目标域信息
        # 获取相对于log_dir的路径
        rel_path = os.path.relpath(txt_file, log_dir)
        match = re.search(r'Office31_(\w)_(\w)_seed\d+\.log/train-\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}\.txt', rel_path)
        if not match:
            continue
        print("matched ", txt_file)
        source, target = match.groups()
        json_file = os.path.join(output_dir, f"{source}_{target}.json")
        
        # 提取准确率
        accuracy = extract_accuracy_from_log(txt_file)
        if accuracy is None:
            print(f"无法从 {txt_file} 提取准确率")
            continue
            
        # 读取或创建JSON文件
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"results": []}
            
        # 添加新的结果
        seed = int(re.search(r'seed(\d+)', rel_path).group(1))
        data["results"].append({
            "run": len(data["results"]) + 1,
            "seed": seed,
            "accuracy": accuracy
        })
        
        # 计算平均准确率
        accuracies = [run["accuracy"] for run in data["results"]]
        data["average_accuracy"] = sum(accuracies) / len(accuracies)
        
        # 保存更新后的JSON文件
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"已更新 {json_file}，准确率: {accuracy}")

def main():
    # 设置目录路径
    log_dir = "logs/mdd/resnet50/Office31"
    output_dir = "output/mdd/resnet50/Office31"
    
    # 更新JSON文件
    update_json_with_accuracy(log_dir, output_dir)
    
    # 计算所有实验的平均结果
    all_files = glob.glob(os.path.join(output_dir, "*.json"))
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
        with open(os.path.join(output_dir, 'overall_results.json'), 'w') as f:
            json.dump({
                'model': 'resnet50',
                'data_name': 'Office31',
                'domain_results': domain_results,
                'overall_average_accuracy': overall_avg
            }, f, indent=4)
        
        print(f'总体平均准确率: {overall_avg:.4f}')
        print(f'详细结果已保存到 {os.path.join(output_dir, "overall_results.json")}')
    else:
        print('没有找到有效的实验结果')

if __name__ == "__main__":
    main() 