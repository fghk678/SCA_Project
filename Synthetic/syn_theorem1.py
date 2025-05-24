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

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

# 导入必要的模块
from CanonicalComponentCCA import CanonicalComponent
from WitnessFunction import WitnessFunction
from Generate_synthetic import GenerateData
from sklearn.manifold import TSNE
from hsicv2 import hsic_gam_torch as hsic

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

# 可视化维度设置
visualize_dim = 5000

def run_theorem1_validation(signal_property, noise_property, data_dimensions):
    """
    运行Theorem 1验证的主要函数
    """
    # 设置随机种子
    random_seed = 7
    torch.manual_seed(random_seed)
    
    # 检查CUDA是否可用
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # 实例化选项
    opt = Opt()
    
    # 生成数据
    data_generate = GenerateData(data_dimensions, opt.batch_size)
    
    mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
    X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(signal_property, noise_property, mixer_property, normalize_mean=True, diff_mixture=True)
    
    # 获取混合矩阵
    A1, A2 = data_generate.get_mixing_matrices()
    A1 = A1.type(Tensor)
    A2 = A2.type(Tensor)
    print("Generating mixing matrix.")
    print(f"Shared Component to Interference Ratio: {scir:.3f}")
    print("Conditions number of mixing matrices: ", torch.linalg.cond(A1), torch.linalg.cond(A2))
    
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
    n_z2 = 1
    
    D = data_dimensions["D"]
    
    # 记录训练过程中的损失
    all_loss_f = []
    all_loss_z = []
    shared_component_dist = []
    
    for epoch in range(n_epochs-1):
        noise_factor = 1.0 - (epoch / n_epochs)
        avg_loss_z = []
        avg_loss_f = []
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
            
            # 计算各种损失
            with torch.no_grad():
                loss_z = loss_fz1 + loss_fz2
                loss_reg = (reg_z1 + reg_z2)/D
                
                s_x = z1(X_1.T).T
                s_y = z2(X_2.T).T
                dist_loss = torch.norm(s_x-s_y)
            
            avg_loss_f.append(loss_f.item())
            avg_loss_z.append(loss_z.item())
            avg_shared_dist.append(dist_loss.item())
            
            print(
                "[Epoch %d of %d]  [z1: %f] [z2: %f] [z: %f] [f: %f] [reg c: %f] [dist: %f]" 
                % (epoch+1, i+1, loss_fz1.item(), loss_fz2.item(), loss_z.item(), loss_f.item(), loss_reg.item(), dist_loss.item())
            )
        
        all_loss_f.append(np.mean(avg_loss_f))
        all_loss_z.append(np.mean(avg_loss_z))
        shared_component_dist.append(np.mean(avg_shared_dist))
    
    return S, C1, C2, z1, z2, A1, A2, X_1, X_2, shared_component_dist

def find_plots(metadata):
    """
    处理和准备绘图数据
    """
    S, C1, C2, z1, z2, A1, A2, X_1, X_2, _ = metadata
    
    D = S.shape[0]
    
    q1 = z1.get_Q_value()
    q2 = z2.get_Q_value()
    theta_1 = (q1 @ A1).detach().cpu()
    theta_2 = (q2 @ A2).detach().cpu()

    print("Theta 1: ", theta_1)
    print("Theta 2: ", theta_2)
    
    if D > 2:
        print("Applying TSNE to shared components. Dimension greater than 2.")
        s_x = z1(X_1.T).T.detach()
        s_y = z2(X_2.T).T.detach()
        
        corig = S[:,:visualize_dim].T.cpu().numpy()
        c1 = s_x[:visualize_dim,].cpu().numpy()
        c2 = s_y[:visualize_dim,].cpu().numpy()

        tsne_shared = TSNE(n_components=2)
        tsne_c = tsne_shared.fit_transform(np.vstack((corig, c1, c2)))
        corig = tsne_c[:visualize_dim, ].T
        c1 = tsne_c[visualize_dim: 2*visualize_dim, ].T
        c2 = tsne_c[2*visualize_dim:, ].T
        
        print("Finished TSNE for shared components.")
    else:
        s_x = z1(X_1.T).T.detach()
        s_y = z2(X_2.T).T.detach()
        
        corig = S[:,:visualize_dim].cpu().numpy()
        c1 = s_x[:visualize_dim,].T.cpu().numpy()
        c2 = s_y[:visualize_dim,].T.cpu().numpy()

    tsne = TSNE(n_components=2)
    tsne_l = tsne.fit_transform(np.vstack((X_1.T[:visualize_dim,].cpu().numpy(), X_2.T[:visualize_dim,].cpu().numpy())))
    domain1 = tsne_l[:visualize_dim, ].T
    domain2 = tsne_l[visualize_dim:, ].T
    print("Finished applying TSNE")

    return corig, c1, c2, domain1, domain2

def plot_and_save_results(corig, c1, c2, domain1, domain2, shared_component_dist):
    """
    绘制和保存结果到output目录
    """
    # 绘制共享组件可视化
    fig = plt.figure(figsize=(16, 7))
    
    # 共享组件绘制
    ax1 = fig.add_subplot(121)
    ax1.scatter(corig[0, :500], corig[1, :500], c='r', alpha=0.5, label='Ground Truth')
    ax1.set_title(r'Ground Truth Shared')
    ax1.set_xlabel(r'$s_1$')
    ax1.set_ylabel(r'$s_2$')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.scatter(c1[0, :500], c1[1, :500], c='b', alpha=0.5, label='View 1')
    ax2.scatter(c2[0, :500], c2[1, :500], c='g', alpha=0.5, label='View 2')
    ax2.set_title(r'Learned Shared')
    ax2.set_xlabel(r'$s_1$')
    ax2.set_ylabel(r'$s_2$')
    ax2.grid(True)
    ax2.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('output/theorem1_shared_components.png')
    plt.close()
    
    # 域绘制
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(domain1[0, :500], domain1[1, :500], c='b', alpha=0.5, label='Domain 1')
    ax.scatter(domain2[0, :500], domain2[1, :500], c='g', alpha=0.5, label='Domain 2')
    ax.set_title('Domain Visualization')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('output/theorem1_domain_visualization.png')
    plt.close()
    
    # 绘制距离变化图
    plt.figure(figsize=(10, 8))
    plt.plot(shared_component_dist)
    plt.title('Shared Component Distance Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.savefig('output/theorem1_distance_trend.png')
    plt.close()

def main():
    print("Executing Theorem 1 Validation")
    
    # 定义数据维度
    data_dimensions = {
        "N": 10**5,  # 样本数
        "D": 2,      # 共享组件向量长度
        "D1": 1,     # 视图1中的私有组件长度
        "D2": 1,     # 视图2中的私有组件长度
        "M1": 3,     # 视图1中的数据长度/混合矩阵
        "M2": 3,     # 视图2中的数据长度/混合矩阵
        "n_epochs": 50
    }
    
    # 测试不同信号和噪声属性
    signal_property = {"mean": 2.5, "std": 2, "dist": "vonmises"}
    noise_property = {"mean1": 1, "std1": 3, "mean2": 0.5, "std2": 3, "dist1": "beta", "dist2": "gamma"}
    
    print("Starting Theorem 1 validation with the following parameters:")
    print(f"Signal property: {signal_property}")
    print(f"Noise property: {noise_property}")
    print(f"Data dimensions: {data_dimensions}")
    
    # 运行模拟
    metadata = run_theorem1_validation(signal_property, noise_property, data_dimensions)
    
    # 准备绘图数据
    corig, c1, c2, domain1, domain2 = find_plots(metadata[:-1])
    shared_component_dist = metadata[-1]
    
    # 绘制和保存结果
    plot_and_save_results(corig, c1, c2, domain1, domain2, shared_component_dist)
    
    print("Theorem 1 validation complete. Results saved to output/ directory.")

if __name__ == "__main__":
    main()