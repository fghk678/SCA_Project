#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from sklearn.manifold import TSNE
import pickle
import time
import math
import random
import traceback
from functools import wraps

# 确保输出目录存在
os.makedirs('output', exist_ok=True)
os.makedirs('output/checkpoints', exist_ok=True)  # 创建保存检查点的目录

# 导入必要的模块
from CanonicalComponentCCA import CanonicalComponent
from WitnessFunction import WitnessFunction
from Generate_synthetic import GenerateData

# 配置参数
from types import SimpleNamespace
opt = {
    "n_epochs": 100,
    "batch_size": 1000,
    "alpha": 0.009,
    "beta": 8e-5,
    "latent_dim": 1,
    "n_critic": 1,
    "lsmooth": 1
}
opt = SimpleNamespace(**opt)
# 可视化维度设置
visualize_dim = 5000

def check_tensor_valid(tensor, name="tensor"):
    """检查张量是否包含NaN或无限值"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    return True

def save_checkpoint(data, filename):
    """保存中间结果到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"检查点已保存到 {filename}")

def load_checkpoint(filename):
    """从文件加载中间结果"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def run_with_retry(func, max_retries=3, *args, **kwargs):
    """
    使用不同随机种子重试函数
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            # 为每次尝试生成不同的随机种子
            if 'random_seed' in kwargs:
                kwargs['random_seed'] = random.randint(1, 10000)
            print(f"尝试运行 {func.__name__}，使用随机种子 {kwargs.get('random_seed', 'default')}")
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            print(f"尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
            traceback.print_exc()
            time.sleep(1)  # 短暂等待
    
    # 所有尝试都失败
    print(f"所有尝试都失败，最后的错误: {str(last_exception)}")
    raise last_exception

def run_anchor_point_simulation(signal_property, noise_property, data_dimensions, anchor_nums=10, random_seed=7, checkpoint_path=None):
    """
    运行带有锚点约束的模拟
    """
    # 设置随机种子
    #Generate data
    torch.manual_seed(random_seed)
    
    # lambdaa = 0.1  #1.2, 0.05, 1e-2 for old f_theta
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # 尝试加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"正在加载检查点: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            return checkpoint
    
    data_generate = GenerateData(data_dimensions, opt.batch_size)
    
    mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
    X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(signal_property, noise_property, mixer_property, normalize_mean=True)
    
    if data_dimensions["D"] >= 3:
        display_type = 3
    else:
        display_type = data_dimensions["D"]
    
    A1, A2 = data_generate.get_mixing_matrices()
    A1 = A1.type(Tensor)
    A2 = A2.type(Tensor)
    print("Conditions number of mixing matrices: ", torch.linalg.cond(A1), torch.linalg.cond(A2))
    
    X_1 = Variable(X_1.type(Tensor))
    X_2 = Variable(X_2.type(Tensor))
    S = Variable(S.type(Tensor))
    
    view1, view2 = data_generate.create_dataloader(X_1, X_2)
    
    z1 = CanonicalComponent(X_1, data_dimensions["D"], data_dimensions["M1"], data_dimensions["N"], Tensor)
    z2 = CanonicalComponent(X_2, data_dimensions["D"], data_dimensions["M2"], data_dimensions["N"], Tensor)
    f = WitnessFunction(data_dimensions["D"], opt.latent_dim)

    if anchor_nums>0:
        print("Multiple anchor points")
        random_indices = np.random.choice(X_1.shape[1], size=anchor_nums, replace=False)
        anchors1 = X_1[:, random_indices]
        anchors2 = X_2[:, random_indices]
    else:
        anc_loss = torch.tensor([0.0]).type(Tensor)
        anchors1 = []
        anchors2 = []
    
    if cuda:
        z1.cuda()
        z2.cuda()
        f.cuda()
    
    # Optimizers
    alpha1 = opt.alpha
    alpha2 = opt.alpha
    beta = opt.beta
    optimizer_z1 = torch.optim.Adam(z1.parameters(), lr=alpha1)
    optimizer_z2 = torch.optim.Adam(z2.parameters(), lr=alpha2)
    optimizer_f = torch.optim.Adam(f.parameters(), lr=beta)
    
    lambdaa = 0.1  #1.2, 0.05, 1e-2 for old f_theta
    lambda_anc = 0.01
    
    n_epochs = data_dimensions["n_epochs"]
    batch_size = opt.batch_size
    lsmooth = 1
    loss_func = nn.BCELoss()
    
    n_critic = 1
    n_z1 = 1
    n_z2 = 1
    
    
    D = data_dimensions["D"]
    all_loss_f = []
    all_loss_z = []
    shared_component_dist = []
    
    # 创建检查点路径
    if not checkpoint_path:
        checkpoint_path = f"output/checkpoints/anchor_{anchor_nums}_seed_{random_seed}_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    
    try:
        for epoch in range(n_epochs-1):
            
            noise_factor = 1.0 - (epoch / n_epochs)
            avg_loss_z = []
            avg_loss_f = []
            avg_shared_dist = []
            for i, (X1, X2) in enumerate(zip(view1, view2)):
                
                ############################Distribution matching ################################
                labels_true = (torch.ones((batch_size, 1) )  - lsmooth * (torch.rand((batch_size, 1)) * 0.2 * noise_factor) ).type(Tensor) 
                labels_false = (torch.zeros((batch_size, 1) ) + lsmooth * (torch.rand((batch_size, 1) ) * 0.2 * noise_factor) ).type(Tensor)
        
                for _ in range(n_critic):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_f.zero_grad()
        
                    # Select one generator to update
                    s_1 = z1(X1).T
                    s_2 = z2(X2).T
                    
                    # 检查输出值是否有效
                    if not check_tensor_valid(s_1, "s_1") or not check_tensor_valid(s_2, "s_2"):
                        raise ValueError(f"检测到无效输出: NaN或无限值")
        
                    # 确保f的输出在[0,1]范围内
                    f_s1 = torch.clamp(f(s_1), 0.0, 1.0)
                    f_s2 = torch.clamp(f(s_2), 0.0, 1.0)
                    
                    # Discriminator loss
                    loss_f = loss_func(f_s1, labels_true) + loss_func(f_s2, labels_false)
                    
                    # 检查损失值
                    if not check_tensor_valid(loss_f, "loss_f"):
                        raise ValueError(f"判别器损失无效: {loss_f.item()}")
        
                    loss_f.backward()
                    optimizer_f.step()
        
                 #######################################################################################
                for _ in range(n_z1):
                    optimizer_z1.zero_grad()
        
                    s_1 = z1(X1).T
                    s_2 = z2(X2).T
                    
                    if not check_tensor_valid(s_1, "s_1") or not check_tensor_valid(s_2, "s_2"):
                        raise ValueError("检测到无效输出: NaN或无限值")
        
                    f_s1 = torch.clamp(f(s_1), 0.0, 1.0)
                    
                    loss_fz1 = loss_func(f_s1, labels_false)
                    reg_z1 =   D * z1.id_loss()
                    if anchor_nums > 0:
                        anc_z1 = z1(anchors1.T)
                        anc_z2 = z2(anchors2.T)
                        if not check_tensor_valid(anc_z1, "anc_z1") or not check_tensor_valid(anc_z2, "anc_z2"):
                            raise ValueError("锚点输出包含无效值")
                        anc_loss = torch.norm(anc_z1 - anc_z2)**2
                        
                    loss_z1 = loss_fz1 + lambdaa * reg_z1 + lambda_anc * anc_loss
                    
                    if not check_tensor_valid(loss_z1, "loss_z1"):
                        raise ValueError(f"z1损失无效: {loss_z1.item()}")
                    
                    loss_z1.backward()
                    optimizer_z1.step()
                
                for _ in range(n_z2):
                    optimizer_z2.zero_grad()
        
                    s_1 = z1(X1).T
                    s_2 = z2(X2).T
                    
                    if not check_tensor_valid(s_1, "s_1") or not check_tensor_valid(s_2, "s_2"):
                        raise ValueError("检测到无效输出: NaN或无限值")
        
                    f_s2 = torch.clamp(f(s_2), 0.0, 1.0)
        
                    loss_fz2 = loss_func(f_s2, labels_true)
                    reg_z2 =   D*z2.id_loss()
                    if anchor_nums > 0:
                        anc_z1 = z1(anchors1.T)
                        anc_z2 = z2(anchors2.T)
                        if not check_tensor_valid(anc_z1, "anc_z1") or not check_tensor_valid(anc_z2, "anc_z2"):
                            raise ValueError("锚点输出包含无效值")
                        anc_loss = torch.norm(anc_z1 - anc_z2)**2
                        
                    loss_z2 = loss_fz2 + lambdaa * reg_z2 + lambda_anc * anc_loss
                    
                    if not check_tensor_valid(loss_z2, "loss_z2"):
                        raise ValueError(f"z2损失无效: {loss_z2.item()}")
                    
                    loss_z2.backward()
                    optimizer_z2.step()
                
                with torch.no_grad():
                    loss_z = loss_fz1 + loss_fz2
                    loss_reg = (reg_z1 + reg_z2)/D
                    
                    s_x = z1(X_1.T).T
                    s_y = z2(X_2.T).T
                    dist_loss = torch.norm(s_x-s_y)
                    
                    # 动态调整锚点损失权重
                    if anchor_nums > 0 and anc_loss.item() > 1.5:
                        lambda_anc = min(lambda_anc * 1.05, 0.1)  # 增加但限制最大值
                    
                avg_loss_f.append( loss_f.item() )
                avg_loss_z.append( loss_z.item() )
                avg_shared_dist.append( dist_loss.item() )
                print(
                    "[Epoch %d of %d]  [z1 loss: %f] [z2 loss: %f] [z loss: %f] [f loss: %f] [anc loss: %f] [reg loss : %f] [%f]" 
                    % (epoch+1, i+1, loss_fz1.item(), loss_fz2.item(), loss_z.item(), loss_f.item(), anc_loss.item(), loss_reg.item(), dist_loss.item() )
                )
                    
            
            all_loss_f.append(np.mean(avg_loss_f))
            all_loss_z.append(np.mean(avg_loss_z))
            shared_component_dist.append(np.mean(avg_shared_dist))
            
            # 每10个epoch保存一次检查点
            if epoch % 10 == 0:
                results = (S, z1, z2, A1, A2, X_1, X_2, anchors1, anchors2, shared_component_dist)
                save_checkpoint(results, checkpoint_path)
                
    except Exception as e:
        print(f"训练中断: {str(e)}")
        print("保存最终检查点...")
        results = (S, z1, z2, A1, A2, X_1, X_2, anchors1, anchors2, shared_component_dist)
        save_checkpoint(results, checkpoint_path)
        raise e
        
    # 最终结果
    results = (S, z1, z2, A1, A2, X_1, X_2, anchors1, anchors2, shared_component_dist)
    save_checkpoint(results, checkpoint_path)
    return results


def main():
    print("执行锚点公式分析")
    print("所有输出将保存在output/目录中")
    
    #维度初始化 M_q>=D, M_q >= (D+D1)cca_loss
    data_dimensions_sym = {
        "N": 10**5, #样本数量
        "D": 3, #共享组件向量长度
        "D1": 1, #私有组件1向量长度
        "D2": 1, #私有组件2向量长度
        "M1": 4, #实际数据视图1/混合矩阵中的数据长度（行数）
        "M2": 4, #实际数据视图2/混合矩阵中的数据长度（行数）
        "n_epochs": 100
    }
    
    # 测试不同信号属性
    signal_property_sym = {"mean": 0, "std": 3, "dist": "laplace"} #这些是参数而非均值和方差，尽管变量名如此
    noise_property_sym = {"mean1": 1, "std1": 3, "mean2": 0.5, "std2": 3, "dist1": "uniform", "dist2": "gamma"}
    
    # 检查点路径
    cp_path_1 = "output/checkpoints/anchor_1.pkl"
    cp_path_2 = "output/checkpoints/anchor_2.pkl"
    cp_path_3 = "output/checkpoints/anchor_3.pkl"
    
    # 运行不同数量锚点的模拟，带有错误恢复
    try:
        print("运行锚点数量为1的模拟...")
        metadata_sym = run_with_retry(
            run_anchor_point_simulation, 
            3,  # 最大重试次数
            signal_property_sym, 
            noise_property_sym, 
            data_dimensions_sym, 
            anchor_nums=1, 
            random_seed=77777,
            checkpoint_path=cp_path_1
        )
        
        print("\n运行锚点数量为2的模拟...")
        metadata_anc2 = run_with_retry(
            run_anchor_point_simulation, 
            3,
            signal_property_sym, 
            noise_property_sym, 
            data_dimensions_sym, 
            anchor_nums=2, 
            random_seed=77777,
            checkpoint_path=cp_path_2
        )
        
        print("\n运行锚点数量为3的模拟...")
        metadata_anc3 = run_with_retry(
            run_anchor_point_simulation, 
            3,
            signal_property_sym, 
            noise_property_sym, 
            data_dimensions_sym, 
            anchor_nums=3, 
            random_seed=3,
            checkpoint_path=cp_path_3
        )
    except Exception as e:
        print(f"一个或多个模拟失败: {str(e)}")
        # 尝试从检查点加载
        metadata_sym = load_checkpoint(cp_path_1)
        metadata_anc2 = load_checkpoint(cp_path_2)
        metadata_anc3 = load_checkpoint(cp_path_3)
        
        # 检查是否有任何模拟完全失败且没有检查点
        if metadata_sym is None or metadata_anc2 is None or metadata_anc3 is None:
            print("无法从检查点恢复必要的数据，无法继续")
            return

    S, z1_sym, z2_sym, _, _, X_1, X_2, _, _, val_sym = metadata_sym
    _, z1_anc2, z2_anc2, _, _, _, _, anc2_1, anc2_2, val_anc2 = metadata_anc2
    _, z1_anc3, z2_anc3, _, _, _, _, anc3_1, anc3_2, val_anc3 = metadata_anc3

    D = S.shape[0]

    if D > 2:
        print("Applying TSNE to shared components. Dimension greater than 2.")
        s_x_sym = z1_sym(X_1.T).T.detach()
        s_y_sym = z2_sym(X_2.T).T.detach()

        s_x_anc2 = z1_anc2(X_1.T).T.detach()
        s_y_anc2 = z2_anc2(X_2.T).T.detach()

        s_x_anc3 = z1_anc3(X_1.T).T.detach()
        s_y_anc3 = z2_anc3(X_2.T).T.detach()
        
        corig = S[:,:visualize_dim].T.cpu().numpy()
        c1_sym = s_x_sym[:visualize_dim,].cpu().numpy()
        c2_sym = s_y_sym[:visualize_dim,].cpu().numpy()

        c1_anc2 = s_x_anc2[:visualize_dim,].cpu().numpy()
        c2_anc2 = s_y_anc2[:visualize_dim,].cpu().numpy()

        c1_anc3 = s_x_anc3[:visualize_dim,].cpu().numpy()
        c2_anc3 = s_y_anc3[:visualize_dim,].cpu().numpy()

        anc2_points  = z1_anc2(anc2_1.T).T.detach().cpu().numpy()
        anc3_points  = z1_anc3(anc3_1.T).T.detach().cpu().numpy()

        # 防止TSNE出现问题
        try:
            tsne_shared = TSNE(n_components=2)
            tsne_c = tsne_shared.fit_transform(np.vstack( (corig,  c1_sym, c2_sym, c1_anc2, c2_anc2, c1_anc3, c2_anc3, anc2_points, anc3_points)  ))
            corig = tsne_c[:visualize_dim, ].T
            c1_sym = tsne_c[visualize_dim: 2*visualize_dim, ].T
            c2_sym = tsne_c[2*visualize_dim: 3*visualize_dim, ].T
            c1_anc2 = tsne_c[3*visualize_dim: 4*visualize_dim, ].T
            c2_anc2 = tsne_c[4*visualize_dim: 5*visualize_dim, ].T
            c1_anc3 = tsne_c[5*visualize_dim: 6*visualize_dim, ].T
            c2_anc3 = tsne_c[6*visualize_dim:7*visualize_dim, ].T
            anc2_points = tsne_c[7*visualize_dim:7*visualize_dim+anc2_points.shape[0] , ].T
            anc3_points = tsne_c[7*visualize_dim+anc2_points.shape[0]:, ].T
        except Exception as e:
            print(f"TSNE出现错误: {str(e)}")
            # 使用PCA作为备选方案
            from sklearn.decomposition import PCA
            print("使用PCA替代TSNE")
            pca = PCA(n_components=2)
            pca_c = pca.fit_transform(np.vstack( (corig,  c1_sym, c2_sym, c1_anc2, c2_anc2, c1_anc3, c2_anc3, anc2_points, anc3_points)  ))
            corig = pca_c[:visualize_dim, ].T
            c1_sym = pca_c[visualize_dim: 2*visualize_dim, ].T
            c2_sym = pca_c[2*visualize_dim: 3*visualize_dim, ].T
            c1_anc2 = pca_c[3*visualize_dim: 4*visualize_dim, ].T
            c2_anc2 = pca_c[4*visualize_dim: 5*visualize_dim, ].T
            c1_anc3 = pca_c[5*visualize_dim: 6*visualize_dim, ].T
            c2_anc3 = pca_c[6*visualize_dim:7*visualize_dim, ].T
            anc2_points = pca_c[7*visualize_dim:7*visualize_dim+anc2_points.shape[0] , ].T
            anc3_points = pca_c[7*visualize_dim+anc2_points.shape[0]:, ].T
        
        print("Finished dimension reduction for shared components.")
        
        
    else:
        s_x_sym = z1_sym(X_1.T).T.detach()
        s_y_sym = z2_sym(X_2.T).T.detach()

        s_x_anc2 = z1_anc2(X_1.T).T.detach()
        s_y_anc2 = z2_anc2(X_2.T).T.detach()

        s_x_anc3 = z1_anc3(X_1.T).T.detach()
        s_y_anc3 = z2_anc3(X_2.T).T.detach()
        
        corig = S[:,:visualize_dim].cpu().numpy()
        c1_sym = s_x_sym[:visualize_dim,].T.cpu().numpy()
        c2_sym = s_y_sym[:visualize_dim,].T.cpu().numpy()

        c1_anc2 = s_x_anc2[:visualize_dim,].T.cpu().numpy()
        c2_anc2 = s_y_anc2[:visualize_dim,].T.cpu().numpy()

        c1_anc3 = s_x_anc3[:visualize_dim,].T.cpu().numpy()
        c2_anc3 = s_y_anc3[:visualize_dim,].T.cpu().numpy()

    # TSNE用于原始数据
    try:
        tsne = TSNE(n_components=2)
        tsne_l = tsne.fit_transform(np.vstack( (X_1.T[:visualize_dim,].cpu().numpy(), X_2.T[:visualize_dim,].cpu().numpy()) ) )
        domain1 = tsne_l[:visualize_dim, ].T
        domain2 = tsne_l[visualize_dim:, ].T
    except Exception as e:
        print(f"原始数据TSNE出现错误: {str(e)}")
        from sklearn.decomposition import PCA
        print("使用PCA替代TSNE")
        pca = PCA(n_components=2)
        pca_l = pca.fit_transform(np.vstack( (X_1.T[:visualize_dim,].cpu().numpy(), X_2.T[:visualize_dim,].cpu().numpy()) ) )
        domain1 = pca_l[:visualize_dim, ].T
        domain2 = pca_l[visualize_dim:, ].T
    
    print("Finished applying dimension reduction")

    #Sometimes the adversarial loss might not converge which might cause the issue below (See limitations section in the paper). 
    #Please ensure by convergence by changing learning rates.

    import matplotlib.gridspec as gridspec
    dims = [0, 1]
    colors = np.arctan2(corig[dims[1]], corig[dims[0]])
    f1size = 20
    f2size = 40
    
    fig = plt.figure(figsize=(24, 12))
    marker_size = 500
    marker_type = "x"
    linewidth = 5

    # Define the outer grid with 2 rows and 2 columns
    outer_grid = gridspec.GridSpec(2, 4, figure=fig)

    # Plot in the first subplot
    ax1 = fig.add_subplot(outer_grid[0, 0])
    ax1.scatter(corig[dims[0]], corig[dims[1]], c=colors, cmap='hsv')
    ax1.set_xlabel(r"$\mathbf{c}$", fontsize=f2size)
    ax1.tick_params(axis='both', which='major', labelsize=f1size)
    ax1.set_title("Ground truth", fontsize=f2size)

    # Define the inner grid for the second subplot
    inner_grid = outer_grid[1, 0].subgridspec(2, 1)

    # Plot in the upper half of the second subplot
    ax2_1 = fig.add_subplot(inner_grid[0])
    ax2_1.scatter(domain1[0], domain1[1], c=colors, cmap='hsv')
    ax2_1.set_ylabel(r"$\mathbf{x}^{(1)}$", fontsize=f2size)
    ax2_1.tick_params(axis='both', which='major', labelsize=f1size)

    # Plot in the lower half of the second subplot
    ax2_2 = fig.add_subplot(inner_grid[1])
    ax2_2.scatter(domain2[0], domain2[1], c=colors, cmap='hsv')
    ax2_2.set_ylabel(r"$\mathbf{x}^{(2)}$", fontsize=f2size)
    ax2_2.tick_params(axis='both', which='major', labelsize=f1size)

    # Plot in the third subplot
    ax3 = fig.add_subplot(outer_grid[0, 1])
    ax3.scatter(c1_sym[dims[0]], c1_sym[dims[1]], c=colors, cmap='hsv')
    ax3.set_xlabel(r"$\widehat{\mathbf{c}}^{(1)}$", fontsize=f2size)
    ax3.tick_params(axis='both', which='major', labelsize=f1size)
    ax3.set_title("No anchors", fontsize=f2size)

    # Plot in the fourth subplot
    ax4 = fig.add_subplot(outer_grid[1, 1])
    ax4.scatter(c2_sym[dims[0]], c2_sym[dims[1]], c=colors, cmap='hsv')
    ax4.set_xlabel(r"$\widehat{\mathbf{c}}^{(2)}$", fontsize=f2size)
    ax4.tick_params(axis='both', which='major', labelsize=f1size)

    ax5 = fig.add_subplot(outer_grid[0, 2])
    ax5.set_xlabel(r"$\widehat{\mathbf{c}}^{(1)}$", fontsize=f2size)
    ax5.scatter(c1_anc2[dims[0]], c1_anc2[dims[1]], c=colors, cmap='hsv')
    ax5.tick_params(axis='both', which='major', labelsize=f1size)
    ax5.scatter(anc2_points[0], anc2_points[1], color='black', label='Anchor Point', s=marker_size, marker= marker_type, linewidths=linewidth)
    ax5.set_title("1 anchors", fontsize=f2size)

    ax6 = fig.add_subplot(outer_grid[1, 2])
    ax6.set_xlabel(r"$\widehat{\mathbf{c}}^{(2)}$", fontsize=f2size)
    ax6.scatter(c2_anc2[dims[0]], c2_anc2[dims[1]], c=colors, cmap='hsv')
    ax6.tick_params(axis='both', which='major', labelsize=f1size)
    ax6.scatter(anc2_points[0], anc2_points[1], color='black', label='Anchor Point', s=marker_size, marker= marker_type, linewidths=linewidth)


    ax7 = fig.add_subplot(outer_grid[0, 3])
    ax7.set_xlabel(r"$\widehat{\mathbf{c}}^{(1)}$", fontsize=f2size)
    ax7.scatter(c1_anc3[dims[0]], c1_anc3[dims[1]], c=colors, cmap='hsv')
    ax7.tick_params(axis='both', which='major', labelsize=f1size)
    ax7.scatter(anc3_points[0], anc3_points[1], color='black', label='Anchor Point', s=marker_size, marker= marker_type, linewidths=linewidth)
    ax7.set_title("3 anchors", fontsize=f2size)

    ax8 = fig.add_subplot(outer_grid[1, 3])
    ax8.set_xlabel(r"$\widehat{\mathbf{c}}^{(2)}$", fontsize=f2size)
    ax8.scatter(c2_anc3[dims[0]], c2_anc3[dims[1]], c=colors, cmap='hsv')
    ax8.tick_params(axis='both', which='major', labelsize=f1size)
    ax8.scatter(anc3_points[0], anc3_points[1], color='black', label='Anchor Point', s=marker_size, marker= marker_type, linewidths=linewidth)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.savefig("theorem5_result.pdf")
    # Show the plot
    plt.show()
    plt.savefig("output/anchor_result.pdf")

if __name__ == "__main__":
    main()