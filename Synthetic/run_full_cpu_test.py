import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

# 强制使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 导入必要的类
try:
    from CanonicalComponentCCA import CanonicalComponent
    from WitnessFunction import WitnessFunction
    from Generate_synthetic import GenerateData
    from hsicv2 import hsic_gam_torch as hsic
    
    print("成功导入所有必要的模块")
    
    # 模拟notebook中的opt设置
    opt = {
        "n_epochs": 5,         # 减少训练轮数
        "batch_size": 100,     # 减少批量大小
        "alpha": 0.009,
        "beta": 8e-5,
        "latent_dim": 1,
        "n_critic": 1,
        "lsmooth": 1
    }
    opt = SimpleNamespace(**opt)
    
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
    
    # 模拟run_simulations函数，但只在CPU上运行
    def run_cpu_simulation(signal_property, noise_property, data_dimensions, anchor_nums=0, random_seed=42):
        print("开始使用CPU进行模拟...")
        
        #设置随机种子
        torch.manual_seed(random_seed)
        
        # 强制使用CPU
        cuda = False
        Tensor = torch.FloatTensor  # 使用CPU tensor
        
        # 数据生成
        data_generate = GenerateData(data_dimensions, opt.batch_size)
        
        mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
        try:
            print("生成数据...")
            X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(
                signal_property, 
                noise_property, 
                mixer_property, 
                normalize_mean=True, 
                diff_mixture=True
            )
            
            if data_dimensions["D"] >= 3:
                display_type = 3
            else:
                display_type = data_dimensions["D"]
            
            A1, A2 = data_generate.get_mixing_matrices()
            A1 = A1.type(Tensor)
            A2 = A2.type(Tensor)
            print("矩阵条件数:", torch.linalg.cond(A1), torch.linalg.cond(A2))
            
            X_1 = Variable(X_1.type(Tensor))
            X_2 = Variable(X_2.type(Tensor))
            S = Variable(S.type(Tensor))
            
            # 锚点处理
            if anchor_nums > 0:
                print("使用多个锚点")
                random_indices = np.random.choice(X_1.shape[1], size=anchor_nums, replace=False)
                anchors1 = X_1[:, random_indices]
                anchors2 = X_2[:, random_indices]
            else:
                anc_loss = torch.tensor([0.0]).type(Tensor)
                anchors1 = []
                anchors2 = []
            
            # 创建数据加载器
            view1, view2 = data_generate.create_dataloader(X_1, X_2)
            
            # 模型初始化
            print("初始化模型...")
            z1 = CanonicalComponent(X_1, data_dimensions["D"], data_dimensions["M1"], data_dimensions["N"], Tensor)
            z2 = CanonicalComponent(X_2, data_dimensions["D"], data_dimensions["M2"], data_dimensions["N"], Tensor)
            f = WitnessFunction(data_dimensions["D"], opt.latent_dim)
            
            z1_p = CanonicalComponent(X_1, data_dimensions["D1"], data_dimensions["M1"], data_dimensions["N"], Tensor)
            z2_p = CanonicalComponent(X_2, data_dimensions["D2"], data_dimensions["M2"], data_dimensions["N"], Tensor)
            
            # 优化器
            alpha1 = opt.alpha
            alpha2 = opt.alpha
            beta = opt.beta
            optimizer_z1 = torch.optim.Adam(z1.parameters(), lr=alpha1)
            optimizer_z2 = torch.optim.Adam(z2.parameters(), lr=alpha1)
            optimizer_f = torch.optim.Adam(f.parameters(), lr=beta)
            
            optimizer_z1_p = torch.optim.Adam(z1_p.parameters(), lr=1e-3)
            optimizer_z2_p = torch.optim.Adam(z2_p.parameters(), lr=1e-3)
            
            # 超参数
            lambdaa = 0.1
            lambdaa_p = 10.0
            lambda_anc = 0.01
            lambda_hsic = 50.0
            
            n_epochs = data_dimensions["n_epochs"]
            batch_size = opt.batch_size
            lsmooth = 1
            loss_func = nn.BCELoss()
            
            n_critic = 1
            n_z1 = 1
            n_z2 = 1
            
            # 训练循环
            print("开始训练...")
            for epoch in range(1, n_epochs + 1):
                try:
                    reg_loss = 0
                    priv_loss = 0
                    z_loss1 = 0
                    z_loss2 = 0
                    f_loss = 0
                    
                    for i, (x1, x2) in enumerate(zip(view1, view2)):
                        # 创建真假标签
                        valid_labels = Variable(Tensor(x1.size(0), 1).fill_(1.0), requires_grad=False)
                        fake_labels = Variable(Tensor(x1.size(0), 1).fill_(0.0), requires_grad=False)
                        
                        # 样本打乱
                        perm = torch.randperm(x1.size(0))
                        x2_shuffle = x2[perm]
                        
                        labels_true = valid_labels
                        labels_false = fake_labels
                        
                        # 训练判别器
                        for _ in range(n_critic):
                            optimizer_f.zero_grad()
                            
                            s_1 = torch.cat([z1(x1), z2(x2)], dim=1)
                            s_2 = torch.cat([z1(x1), z2(x2_shuffle)], dim=1)
                            
                            # 判别器损失
                            loss_f = loss_func(f(s_1), labels_true) + loss_func(f(s_2), labels_false)
                            f_loss += loss_f.item()
                            
                            loss_f.backward(retain_graph=True)
                            optimizer_f.step()
                        
                        # 训练view1编码器
                        for _ in range(n_z1):
                            optimizer_z1.zero_grad()
                            optimizer_z1_p.zero_grad()
                            
                            anc_l = torch.tensor([0.0]).type(Tensor)
                            
                            s_1 = torch.cat([z1(x1), z2(x2)], dim=1)
                            s_2 = torch.cat([z1(x1), z2(x2_shuffle)], dim=1)
                            
                            # 锚点损失
                            if anchor_nums > 0:
                                for j in range(anchor_nums):
                                    h1 = z1(anchors1[:, j].reshape(-1, 1))
                                    h2 = z2(anchors2[:, j].reshape(-1, 1))
                                    anc_l += torch.sum((h1 - h2) ** 2)
                                anc_l = anc_l / anchor_nums
                            
                            # HSIC损失，确保view1私有部分与共享部分正交
                            hsic_shared = hsic(z1(x1).detach().reshape(-1, data_dimensions["D"]), z1_p(x1).reshape(-1, data_dimensions["D1"]))
                            
                            # view1编码器损失
                            loss_z1 = loss_func(f(s_1), valid_labels) + lambdaa * loss_func(f(s_2), fake_labels) + lambda_anc * anc_l + lambda_hsic * hsic_shared
                            
                            z_loss1 += loss_z1.item()
                            
                            loss_z1.backward(retain_graph=True)
                            optimizer_z1.step()
                            optimizer_z1_p.step()
                        
                        # 训练view2编码器
                        for _ in range(n_z2):
                            optimizer_z2.zero_grad()
                            optimizer_z2_p.zero_grad()
                            
                            s_1 = torch.cat([z1(x1), z2(x2)], dim=1)
                            s_2 = torch.cat([z1(x1), z2(x2_shuffle)], dim=1)
                            
                            # HSIC损失，确保view2私有部分与共享部分正交
                            hsic_shared = hsic(z2(x2).detach().reshape(-1, data_dimensions["D"]), z2_p(x2).reshape(-1, data_dimensions["D2"]))
                            
                            # view2编码器损失
                            loss_z2 = loss_func(f(s_1), valid_labels) + lambdaa * loss_func(f(s_2), fake_labels) + lambda_hsic * hsic_shared
                            
                            z_loss2 += loss_z2.item()
                            
                            loss_z2.backward(retain_graph=True)
                            optimizer_z2.step()
                            optimizer_z2_p.step()
                    
                    # 输出训练信息
                    print(f"[Epoch {epoch} of {n_epochs}] [z1: {z_loss1:.6f}] [z2: {z_loss2:.6f}] [z: {z_loss1 + z_loss2:.6f}] [f: {f_loss:.6f}] [anc: {anc_l.item():.6f}]")
                
                except Exception as e:
                    print(f"训练过程中出错: {e}")
                    return None
            
            print("训练完成，准备返回结果...")
            # 简化的结果返回
            return {"success": True, "message": "CPU模拟成功完成"}
            
        except Exception as e:
            print(f"模拟过程中出错: {e}")
            return {"success": False, "message": f"错误: {str(e)}"}
    
    # 运行CPU版本的模拟
    result = run_cpu_simulation(
        signal_property_test, 
        noise_property_test, 
        data_dimensions_test, 
        anchor_nums=3,  # 使用3个锚点，模拟原始错误情况
        random_seed=42
    )
    
    # 输出结果
    if result and result["success"]:
        print("成功: " + result["message"])
    else:
        print("失败: " + (result["message"] if result else "未知错误"))
    
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的模块都可用") 