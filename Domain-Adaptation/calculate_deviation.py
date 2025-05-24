import os
import json
import glob
import numpy as np

def calculate_standard_deviation(json_file):
    """计算JSON文件中结果的标准差"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 检查是否有results字段
        if 'results' not in data:
            print(f"文件 {json_file} 中没有找到results字段")
            return None
            
        # 获取所有准确率
        accuracies = [run['accuracy'] for run in data['results']]
        if len(accuracies) < 5:
            print(f"文件 {json_file} 中结果数量不足5个")
            return None
        if "standard_deviation" in data:
            print(f"文件 {json_file} 中已经存在standard_deviation字段")
            return None
        
        # 计算标准差
        std_dev = np.std(accuracies)
        
        # 更新JSON数据
        if "average_accuracy" not in data:
            data['average_accuracy'] = float(np.mean(accuracies))
        data['standard_deviation'] = float(std_dev)
        
        # 保存更新后的文件
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"已更新 {json_file}，标准差: {std_dev:.4f}")
        return std_dev
        
    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {str(e)}")
        return None

def process_all_json_files(base_dir):
    """处理指定目录下的所有JSON文件"""
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(base_dir, "**/*.json"), recursive=True)
    
    # 排除overall_results.json文件
    json_files = [f for f in json_files if not f.endswith('overall_results.json')]
    
    total_files = len(json_files)
    processed_files = 0
    
    print(f"找到 {total_files} 个JSON文件需要处理")
    
    # 处理每个文件
    for json_file in json_files:
        if calculate_standard_deviation(json_file) is not None:
            processed_files += 1
            
    print(f"处理完成！成功更新了 {processed_files}/{total_files} 个文件")

def main():
    # 设置基础目录
    base_dir = "output"
    
    # 处理所有JSON文件
    process_all_json_files(base_dir)

if __name__ == "__main__":
    main() 