import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

# 强制使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 导入必要的类，如果能找到的话
try:
    from CanonicalComponentCCA import CanonicalComponent
    from WitnessFunction import WitnessFunction
    from Generate_synthetic import GenerateData
    from hsicv2 import hsic_gam_torch as hsic
    
    print("成功导入所有必要的模块")
    
    # 创建一个简化版的数据维度字典
    data_dimensions_test = {
        "N": 10**4,  # 减少样本数量
        "D": 2,      # 减少维度
        "D1": 1,
        "D2": 1,
        "M1": 3,
        "M2": 3,
        "n_epochs": 5  # 减少训练轮数
    }
    
    # 创建简化的信号和噪声属性
    signal_property_test = {"mean": 0, "std": 1, "dist": "normal"}
    noise_property_test = {"mean1": 0, "std1": 1, "mean2": 0, "std2": 1, "dist1": "normal", "dist2": "normal"}
    
    # 简化版的运行函数
    def run_test_simulation():
        # 设置随机种子
        torch.manual_seed(42)
        
        # 强制使用CPU
        Tensor = torch.FloatTensor
        
        print("开始生成数据...")
        # 创建GenerateData实例
        batch_size = 100
        data_generate = GenerateData(data_dimensions_test, batch_size)
        
        # 生成CCA数据
        mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
        try:
            X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(
                signal_property_test, 
                noise_property_test, 
                mixer_property, 
                normalize_mean=True, 
                diff_mixture=True
            )
            print("成功生成数据")
            print(f"X_1形状: {X_1.shape}, X_2形状: {X_2.shape}")
            
            # 将数据转换为变量
            X_1 = Variable(X_1)
            X_2 = Variable(X_2)
            S = Variable(S)
            
            print("成功完成基本数据处理")
            return True
        except Exception as e:
            print(f"在生成数据过程中出错: {e}")
            return False
    
    # 运行测试
    success = run_test_simulation()
    print(f"测试{'成功' if success else '失败'}")
    
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的模块都可用") 