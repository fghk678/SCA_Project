#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import random
from tqdm import tqdm

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

# 导入必要的模块
from CanonicalComponentCCA import CanonicalComponent
from WitnessFunction import WitnessFunction
from Generate_synthetic import GenerateData
from sklearn.manifold import TSNE

# 配置参数
class Opt:
    def __init__(self):
        self.n_epochs = 100
        self.batch_size = 1000
        self.alpha = 0.009
        self.beta = 8e-5
        self.latent_dim = 1
        self.n_critic = 1
        self.lsmooth = 1

def run_gan_convergence_test(signal_property, noise_property, data_dimensions, num_tests=10):
    """
    运行多次测试分析GAN收敛成功率
    """
    # 检查CUDA是否可用
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # 实例化选项
    opt = Opt()
    
    # 准备记录各种指标的列表
    convergence_results = []
    distance_trends = []
    final_distances = []
    
    # 运行多次测试
    for test_idx in tqdm(range(num_tests), desc="Running GAN convergence tests"):
        # 设置随机种子
        random_seed = test_idx + 1
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 生成数据
        data_generate = GenerateData(data_dimensions, opt.batch_size)
        
        mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
        try:
            X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(signal_property, noise_property, mixer_property, normalize_mean=True, diff_mixture=True)
            
            # 获取混合矩阵
            A1, A2 = data_generate.get_mixing_matrices()
            A1 = A1.type(Tensor)
            A2 = A2.type(Tensor)
            
            X_1 = Variable(X_1.type(Tensor))
            X_2 = Variable(X_2.type(Tensor))
            S = Variable(S.type(Tensor))

            view1, view2 = data_generate.create_dataloader(X_1, X_2)
            
            # 初始化组件
            z1 = CanonicalComponent(X_1, data_dimensions["D"], data_dimensions["M1"], data_dimensions["N"], Tensor)
            z2 = CanonicalComponent(X_2, data_dimensions["D"], data_dimensions["M2"], data_dimensions["N"], Tensor)
            f = WitnessFunction(data_dimensions["D"], opt.latent_dim)
            
            if cuda:
                z1.cuda()
                z2.cuda()
                f.cuda()
            
            # 优化器
            alpha1 = opt.alpha
            alpha2 = opt.alpha
            beta = opt.beta
            optimizer_z1 = torch.optim.Adam(z1.parameters(), lr=alpha1)
            optimizer_z2 = torch.optim.Adam(z2.parameters(), lr=alpha1)
            optimizer_f = torch.optim.Adam(f.parameters(), lr=beta)
            
            # 超参数设置
            lambdaa = 0.1
            
            # 训练参数
            n_epochs = data_dimensions["n_epochs"]
            batch_size = opt.batch_size
            lsmooth = 1
            loss_func = nn.BCEWithLogitsLoss()
            
            n_critic = 1
            n_z1 = 1
            
            D = data_dimensions["D"]
            
            # 记录训练过程中的损失
            shared_component_dist = []
            
            # 训练模型
            for epoch in range(n_epochs):
                noise_factor = 1.0 - (epoch / n_epochs)
                avg_shared_dist = []
                
                for i, (X1, X2) in enumerate(zip(view1, view2)):
                    # 标签生成
                    labels_true = (torch.ones((batch_size, 1)) - lsmooth * (torch.rand((batch_size, 1)) * 0.2 * noise_factor)).type(Tensor) 
                    labels_false = (torch.zeros((batch_size, 1)) + lsmooth * (torch.rand((batch_size, 1)) * 0.2 * noise_factor)).type(Tensor)
            
                    # 训练判别器
                    for _ in range(n_critic):
                        optimizer_f.zero_grad()
                        
                        s_1 = z1(X1).T
                        s_2 = z2(X2).T
                        
                        loss_f = loss_func(f(s_1), labels_true) + loss_func(f(s_2), labels_false)
                        
                        loss_f.backward()
                        optimizer_f.step()
                    
                    # 训练生成器
                    for _ in range(n_z1):
                        optimizer_z1.zero_grad()
                        optimizer_z2.zero_grad()
                        
                        s_1 = z1(X1).T
                        s_2 = z2(X2).T
                        
                        loss_fz1 = loss_func(f(s_1), labels_false)
                        reg_z1 = D * z1.id_loss()
                        loss_fz2 = loss_func(f(s_2), labels_true)
                        reg_z2 = D * z2.id_loss()
                        
                        loss_z1 = loss_fz1 + loss_fz2 + lambdaa * reg_z1 + lambdaa * reg_z2
                        
                        loss_z1.backward()
                        optimizer_z1.step()
                        optimizer_z2.step()
                    
                    # 计算共享组件的距离
                    with torch.no_grad():
                        s_x = z1(X_1.T).T
                        s_y = z2(X_2.T).T
                        dist_loss = torch.norm(s_x-s_y).item()
                    
                    avg_shared_dist.append(dist_loss)
                
                shared_component_dist.append(np.mean(avg_shared_dist))
            
            # 检查收敛情况
            final_distance = shared_component_dist[-1]
            final_distances.append(final_distance)
            
            # 计算收敛率指标 (如果最终距离小于阈值，则认为收敛成功)
            threshold = 0.5  # 可以根据实际情况调整
            convergence_success = final_distance < threshold
            convergence_results.append(convergence_success)
            
            # 记录距离变化趋势
            distance_trends.append(shared_component_dist)
            
            print(f"Test {test_idx+1}: Final distance = {final_distance:.4f}, Convergence = {convergence_success}")
            
        except Exception as e:
            print(f"Error in test {test_idx+1}: {e}")
            convergence_results.append(False)
            final_distances.append(float('inf'))
            distance_trends.append([])
    
    # 计算总体成功率
    success_rate = np.mean(convergence_results) * 100
    
    return convergence_results, distance_trends, final_distances, success_rate

def plot_and_save_convergence_results(convergence_results, distance_trends, final_distances, success_rate):
    """
    绘制和保存GAN收敛分析结果
    """
    # 绘制成功率
    plt.figure(figsize=(10, 6))
    plt.bar(['Success', 'Failure'], [success_rate, 100-success_rate], color=['green', 'red'])
    plt.title(f'GAN Convergence Success Rate: {success_rate:.1f}%')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    for i, v in enumerate([success_rate, 100-success_rate]):
        plt.text(i, v+1, f"{v:.1f}%", ha='center')
    plt.savefig('output/gan_success_rate.png')
    plt.close()
    
    # 绘制最终距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(final_distances, bins=10, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(final_distances), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(final_distances):.4f}')
    plt.title('Distribution of Final Distances')
    plt.xlabel('Final Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('output/gan_final_distance_distribution.png')
    plt.close()
    
    # 绘制所有测试的距离变化趋势
    plt.figure(figsize=(12, 8))
    for i, trend in enumerate(distance_trends):
        if len(trend) > 0:  # 只绘制有效的趋势
            plt.plot(trend, alpha=0.3, color='blue')
    
    # 绘制平均趋势
    valid_trends = [t for t in distance_trends if len(t) > 0]
    if valid_trends:
        # 找到最短的趋势长度
        min_length = min(len(t) for t in valid_trends)
        # 截断所有趋势到最短长度
        truncated_trends = [t[:min_length] for t in valid_trends]
        # 计算平均趋势
        mean_trend = np.mean(truncated_trends, axis=0)
        plt.plot(mean_trend, linewidth=2, color='red', label='Mean Trend')
    
    plt.title('Distance Trends Across All Tests')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig('output/gan_distance_trends.png')
    plt.close()

def run_parameter_sweep(base_data_dimensions, num_tests=5):
    """
    对不同参数进行扫描，分析它们对GAN收敛的影响
    """
    # 不同的信号和噪声属性组合
    signal_properties = [
        {"mean": 2.5, "std": 2, "dist": "vonmises"},
        {"mean": 0, "std": 1, "dist": "normal"},
        {"mean": 1, "std": 1.5, "dist": "gamma"}
    ]
    
    noise_properties = [
        {"mean1": 1, "std1": 3, "mean2": 0.5, "std2": 3, "dist1": "beta", "dist2": "gamma"},
        {"mean1": 0, "std1": 1, "mean2": 0, "std2": 1, "dist1": "normal", "dist2": "normal"},
        {"mean1": 2, "std1": 2, "mean2": 2, "std2": 2, "dist1": "vonmises", "dist2": "vonmises"}
    ]
    
    # 存储结果
    results = []
    
    # 对每个参数组合进行测试
    for signal_idx, signal_property in enumerate(signal_properties):
        for noise_idx, noise_property in enumerate(noise_properties):
            print(f"Testing parameter combination: Signal {signal_idx+1}, Noise {noise_idx+1}")
            print(f"Signal: {signal_property}")
            print(f"Noise: {noise_property}")
            
            # 运行当前参数下的测试
            convergence_results, _, final_distances, success_rate = run_gan_convergence_test(
                signal_property, noise_property, base_data_dimensions, num_tests
            )
            
            # 记录结果
            results.append({
                'signal_property': signal_property,
                'noise_property': noise_property,
                'success_rate': success_rate,
                'avg_final_distance': np.mean(final_distances),
                'convergence_results': convergence_results
            })
    
    return results

def plot_parameter_sweep_results(results):
    """
    绘制参数扫描结果
    """
    # 提取成功率和平均最终距离
    success_rates = [r['success_rate'] for r in results]
    avg_distances = [r['avg_final_distance'] for r in results]
    
    # 创建标签
    labels = [f"S{i//3+1}N{i%3+1}" for i in range(len(results))]
    
    # 绘制成功率
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, success_rates, color='skyblue')
    plt.title('GAN Convergence Success Rate for Different Parameter Combinations')
    plt.xlabel('Parameter Combination')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.savefig('output/parameter_sweep_success_rates.png')
    plt.close()
    
    # 绘制平均最终距离
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, avg_distances, color='lightgreen')
    plt.title('Average Final Distance for Different Parameter Combinations')
    plt.xlabel('Parameter Combination')
    plt.ylabel('Average Final Distance')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.savefig('output/parameter_sweep_avg_distances.png')
    plt.close()
    
    # 保存详细结果到文本文件
    with open('output/parameter_sweep_results.txt', 'w') as f:
        f.write("Parameter Sweep Results:\n\n")
        for i, r in enumerate(results):
            f.write(f"Combination {i+1}: S{i//3+1}N{i%3+1}\n")
            f.write(f"Signal property: {r['signal_property']}\n")
            f.write(f"Noise property: {r['noise_property']}\n")
            f.write(f"Success rate: {r['success_rate']:.2f}%\n")
            f.write(f"Average final distance: {r['avg_final_distance']:.4f}\n")
            f.write(f"Individual test results: {r['convergence_results']}\n\n")

def main():
    print("Executing Success Rate of GAN Convergence Analysis")
    
    # 基础数据维度设置
    base_data_dimensions = {
        "N": 10**5,  # 样本数
        "D": 2,      # 共享组件向量长度
        "M1": 3,     # 视图1中的数据长度/混合矩阵
        "M2": 3,     # 视图2中的数据长度/混合矩阵
        "n_epochs": 30  # 减少epochs以加快测试
    }
    
    # 运行方式选择 - 单一参数测试或参数扫描
    run_mode = "single"  # "single" 或 "sweep"
    
    if run_mode == "single":
        # 单一参数设置下的收敛分析
        signal_property = {"mean": 2.5, "std": 2, "dist": "vonmises"}
        noise_property = {"mean1": 1, "std1": 3, "mean2": 0.5, "std2": 3, "dist1": "beta", "dist2": "gamma"}
        
        print("Starting GAN convergence analysis with the following parameters:")
        print(f"Signal property: {signal_property}")
        print(f"Noise property: {noise_property}")
        print(f"Data dimensions: {base_data_dimensions}")
        
        # 运行10次测试
        num_tests = 10
        convergence_results, distance_trends, final_distances, success_rate = run_gan_convergence_test(
            signal_property, noise_property, base_data_dimensions, num_tests
        )
        
        # 绘制和保存结果
        plot_and_save_convergence_results(convergence_results, distance_trends, final_distances, success_rate)
        
        print(f"GAN convergence analysis complete. Success rate: {success_rate:.2f}%")
        print("Results saved to output/ directory.")
        
    elif run_mode == "sweep":
        # 参数扫描分析
        print("Starting parameter sweep analysis...")
        
        # 运行参数扫描 (每个组合5次测试)
        num_tests_per_combo = 5
        sweep_results = run_parameter_sweep(base_data_dimensions, num_tests_per_combo)
        
        # 绘制和保存结果
        plot_parameter_sweep_results(sweep_results)
        
        print("Parameter sweep analysis complete. Results saved to output/ directory.")

if __name__ == "__main__":
    main()