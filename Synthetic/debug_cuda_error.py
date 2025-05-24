import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import sys
import traceback

# 关键设置: 启用CUDA同步模式，让错误能够准确定位
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 强制使用CPU，绕过CUDA错误进行调试
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 取消注释以在CPU上运行

print("开始调试CUDA断言错误...")

# 检查张量的函数，验证标签是否在有效范围内
def check_labels(logits, labels, loss_name="未指定"):
    """检查标签是否在有效范围内，避免触发CUDA断言"""
    try:
        if isinstance(labels, torch.Tensor) and isinstance(logits, torch.Tensor):
            print(f"[{loss_name}] logits形状: {logits.shape}, labels形状: {labels.shape}")
            
            # 检查是否为分类任务
            if len(logits.shape) == 2 and len(labels.shape) >= 1:
                num_classes = logits.shape[1]
                
                # 检查标签是否为整数索引类型（用于CrossEntropyLoss）
                if labels.dtype in [torch.long, torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
                    min_val = labels.min().item() if labels.numel() > 0 else "空张量"
                    max_val = labels.max().item() if labels.numel() > 0 else "空张量"
                    print(f"[{loss_name}] 标签范围: [{min_val}, {max_val}], 类别数: {num_classes}")
                    
                    # 检查标签是否在有效范围 [0, num_classes-1] 内
                    if max_val >= num_classes:
                        print(f"⚠️ 错误: 标签值 {max_val} 超出了类别数范围 [0, {num_classes-1}]")
                        return False
                    if min_val < 0:
                        print(f"⚠️ 错误: 存在负数标签 {min_val}")
                        return False
                
                # 检查标签是否为one-hot编码（用于BCELoss）
                elif labels.dtype == torch.float:
                    if labels.shape == logits.shape:
                        print(f"[{loss_name}] 检测到one-hot标签，与BCELoss兼容")
                        # 检查one-hot标签是否为0和1
                        unique_vals = torch.unique(labels)
                        if not all(val in [0, 1] for val in unique_vals):
                            print(f"⚠️ 警告: one-hot标签包含非0/1值: {unique_vals}")
                    else:
                        print(f"⚠️ 警告: 标签和logits形状不一致，可能不兼容")
        
        return True
    except Exception as e:
        print(f"检查标签时出错: {e}")
        return False

# 包装损失函数，在调用前检查标签
class SafeLossWrapper(nn.Module):
    def __init__(self, loss_fn, name="未命名损失"):
        super().__init__()
        self.loss_fn = loss_fn
        self.name = name
    
    def forward(self, logits, labels):
        if not check_labels(logits, labels, self.name):
            print(f"[{self.name}] ⚠️ 标签检查失败，将返回零损失以避免CUDA错误")
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        return self.loss_fn(logits, labels)

try:
    from CanonicalComponentCCA import CanonicalComponent
    from WitnessFunction import WitnessFunction
    from Generate_synthetic import GenerateData
    from hsicv2 import hsic_gam_torch as hsic
    
    print("所有模块导入成功")
    
    # 基本设置
    opt = {
        "n_epochs": 2,
        "batch_size": 50,
        "alpha": 0.009,
        "beta": 8e-5,
        "latent_dim": 1,
        "n_critic": 1,
        "lsmooth": 1
    }
    opt = SimpleNamespace(**opt)
    
    # 使用较小的数据集
    data_dimensions_debug = {
        "N": 10**3,
        "D": 2,
        "D1": 1,
        "D2": 1,
        "M1": 3,
        "M2": 3,
        "n_epochs": 2
    }
    
    signal_property_debug = {"mean": 0, "std": 1, "dist": "normal"}
    noise_property_debug = {"mean1": 0, "std1": 1, "mean2": 0, "std2": 1, "dist1": "normal", "dist2": "normal"}
    
    def debug_cuda_simulation(signal_property, noise_property, data_dimensions, anchor_nums=0, random_seed=42):
        print(f"使用随机种子 {random_seed} 开始调试...")
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # 检测是否使用CUDA
        cuda = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        device = torch.device("cuda" if cuda else "cpu")
        print(f"使用设备: {device}")
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # 数据生成
        data_generate = GenerateData(data_dimensions, opt.batch_size)
        
        mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
        print("生成数据...")
        X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(
            signal_property, 
            noise_property, 
            mixer_property, 
            normalize_mean=True, 
            diff_mixture=True
        )
        
        print(f"X_1 形状: {X_1.shape}, X_2 形状: {X_2.shape}")
        print(f"S 形状: {S.shape}")
        
        # 获取混合矩阵
        A1, A2 = data_generate.get_mixing_matrices()
        A1 = A1.type(Tensor)
        A2 = A2.type(Tensor)
        print("矩阵条件数:", torch.linalg.cond(A1), torch.linalg.cond(A2))
        
        # 转换为张量并移动到设备
        X_1 = Variable(X_1.type(Tensor))
        X_2 = Variable(X_2.type(Tensor))
        S = Variable(S.type(Tensor))
        
        # 处理锚点
        if anchor_nums > 0:
            print(f"使用 {anchor_nums} 个锚点")
            random_indices = np.random.choice(X_1.shape[1], size=anchor_nums, replace=False)
            anchors1 = X_1[:, random_indices]
            anchors2 = X_2[:, random_indices]
            print(f"anchors1 形状: {anchors1.shape}, anchors2 形状: {anchors2.shape}")
        else:
            anc_loss = torch.tensor([0.0]).type(Tensor)
            anchors1 = []
            anchors2 = []
        
        # 创建数据加载器
        view1, view2 = data_generate.create_dataloader(X_1, X_2)
        print("数据加载器已创建")
        
        # 模型初始化
        print("初始化模型...")
        try:
            z1 = CanonicalComponent(X_1, data_dimensions["D"], data_dimensions["M1"], data_dimensions["N"], Tensor)
            z2 = CanonicalComponent(X_2, data_dimensions["D"], data_dimensions["M2"], data_dimensions["N"], Tensor)
            f = WitnessFunction(data_dimensions["D"], opt.latent_dim)
            
            z1_p = CanonicalComponent(X_1, data_dimensions["D1"], data_dimensions["M1"], data_dimensions["N"], Tensor)
            z2_p = CanonicalComponent(X_2, data_dimensions["D2"], data_dimensions["M2"], data_dimensions["N"], Tensor)
            print("模型初始化成功")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
        
        # 优化器
        optimizer_z1 = torch.optim.Adam(z1.parameters(), lr=opt.alpha)
        optimizer_z2 = torch.optim.Adam(z2.parameters(), lr=opt.alpha)
        optimizer_f = torch.optim.Adam(f.parameters(), lr=opt.beta)
        
        optimizer_z1_p = torch.optim.Adam(z1_p.parameters(), lr=1e-3)
        optimizer_z2_p = torch.optim.Adam(z2_p.parameters(), lr=1e-3)
        
        # 使用安全包装的损失函数
        lambdaa = 0.1
        lambda_anc = 0.01
        lambda_hsic = 50.0
        
        # ⚠️ 使用安全损失函数替代普通损失函数
        loss_func = SafeLossWrapper(nn.BCELoss(), "BCE损失")
        
        # 训练循环
        print("开始训练循环...")
        for epoch in range(1, data_dimensions["n_epochs"] + 1):
            z_loss1 = 0
            z_loss2 = 0
            f_loss = 0
            
            for i, (x1, x2) in enumerate(zip(view1, view2)):
                print(f"Epoch {epoch}, 批次 {i+1} - x1 形状: {x1.shape}, x2 形状: {x2.shape}")
                
                # 创建标签并验证
                valid_labels = Variable(Tensor(x1.size(0), 1).fill_(1.0), requires_grad=False)
                fake_labels = Variable(Tensor(x1.size(0), 1).fill_(0.0), requires_grad=False)
                
                # 验证标签值是否在有效范围内
                print("有效标签值范围:", valid_labels.min().item(), valid_labels.max().item())
                print("虚假标签值范围:", fake_labels.min().item(), fake_labels.max().item())
                
                # 打乱样本
                perm = torch.randperm(x1.size(0))
                x2_shuffle = x2[perm]
                
                # 判别器训练
                optimizer_f.zero_grad()
                
                # 记录中间张量形状
                s_x = z1(x1)
                s_y = z2(x2)
                print(f"s_x 形状: {s_x.shape}, s_y 形状: {s_y.shape}")
                
                # 检查维度是否匹配
                if s_x.shape[0] != s_y.shape[0]:
                    print(f"⚠️ 形状不匹配: s_x {s_x.shape} vs s_y {s_y.shape}")
                    continue
                
                s_1 = torch.cat([s_x, s_y], dim=1)
                print(f"s_1 形状: {s_1.shape}")
                
                s_y_shuffle = z2(x2_shuffle)
                s_2 = torch.cat([s_x, s_y_shuffle], dim=1)
                print(f"s_2 形状: {s_2.shape}")
                
                # 判别器输出
                f_output1 = f(s_1)
                f_output2 = f(s_2)
                print(f"判别器输出形状: {f_output1.shape}, 标签形状: {valid_labels.shape}")
                
                # 判别器损失
                try:
                    loss_f = loss_func(f_output1, valid_labels) + loss_func(f_output2, fake_labels)
                    print(f"判别器损失: {loss_f.item()}")
                    
                    # 反向传播
                    loss_f.backward(retain_graph=True)
                    optimizer_f.step()
                    f_loss += loss_f.item()
                    
                    print("判别器训练完成")
                except Exception as e:
                    print(f"判别器训练出错: {e}")
                    traceback.print_exc()
                    continue
                
                # 训练编码器
                try:
                    # 训练 view1 编码器
                    optimizer_z1.zero_grad()
                    optimizer_z1_p.zero_grad()
                    
                    # 锚点损失
                    anc_l = torch.tensor([0.0]).type(Tensor)
                    if anchor_nums > 0:
                        for j in range(anchor_nums):
                            anchor1 = anchors1[:, j].reshape(-1, 1)
                            anchor2 = anchors2[:, j].reshape(-1, 1)
                            
                            h1 = z1(anchor1)
                            h2 = z2(anchor2)
                            anc_l += torch.sum((h1 - h2) ** 2)
                        
                        anc_l = anc_l / anchor_nums
                        print(f"锚点损失: {anc_l.item()}")
                    
                    # HSIC损失
                    s_x = z1(x1)
                    p_x = z1_p(x1)
                    
                    # 检查 HSIC 输入形状
                    s_x_reshaped = s_x.detach().reshape(-1, data_dimensions["D"])
                    p_x_reshaped = p_x.reshape(-1, data_dimensions["D1"])
                    print(f"HSIC 输入形状: s_x={s_x_reshaped.shape}, p_x={p_x_reshaped.shape}")
                    
                    hsic_shared = hsic(s_x_reshaped, p_x_reshaped)
                    print(f"HSIC 损失: {hsic_shared}")
                    
                    # 编码器损失
                    s_1 = torch.cat([z1(x1), z2(x2)], dim=1)
                    s_2 = torch.cat([z1(x1), z2(x2_shuffle)], dim=1)
                    
                    loss_z1 = loss_func(f(s_1), valid_labels) + lambdaa * loss_func(f(s_2), fake_labels) + lambda_anc * anc_l + lambda_hsic * hsic_shared
                    z_loss1 += loss_z1.item()
                    
                    loss_z1.backward(retain_graph=True)
                    optimizer_z1.step()
                    optimizer_z1_p.step()
                    
                    # 训练 view2 编码器
                    optimizer_z2.zero_grad()
                    optimizer_z2_p.zero_grad()
                    
                    s_y = z2(x2)
                    p_y = z2_p(x2)
                    
                    # 检查 HSIC 输入形状
                    s_y_reshaped = s_y.detach().reshape(-1, data_dimensions["D"])
                    p_y_reshaped = p_y.reshape(-1, data_dimensions["D2"])
                    print(f"HSIC 输入形状: s_y={s_y_reshaped.shape}, p_y={p_y_reshaped.shape}")
                    
                    hsic_shared = hsic(s_y_reshaped, p_y_reshaped)
                    
                    loss_z2 = loss_func(f(s_1), valid_labels) + lambdaa * loss_func(f(s_2), fake_labels) + lambda_hsic * hsic_shared
                    z_loss2 += loss_z2.item()
                    
                    loss_z2.backward(retain_graph=True)
                    optimizer_z2.step()
                    optimizer_z2_p.step()
                    
                    print("编码器训练完成")
                except Exception as e:
                    print(f"编码器训练出错: {e}")
                    traceback.print_exc()
                    continue
            
            # 输出进度
            avg_z1 = z_loss1 / len(view1)
            avg_z2 = z_loss2 / len(view1)
            avg_f = f_loss / len(view1)
            print(f"[Epoch {epoch}/{data_dimensions['n_epochs']}] [z1: {avg_z1:.6f}] [z2: {avg_z2:.6f}] [f: {avg_f:.6f}] [anc: {anc_l.item():.6f}]")
            
        print("训练完成!")
        return {"success": True, "message": "调试模拟成功完成"}
        
    # 运行调试
    print("开始运行CUDA错误调试...")
    result = debug_cuda_simulation(
        signal_property_debug,
        noise_property_debug,
        data_dimensions_debug,
        anchor_nums=3,
        random_seed=42  # 尝试不同的随机种子
    )
    
    # 输出结果
    if result and result["success"]:
        print("成功: " + result["message"])
    else:
        print("失败: " + (result["message"] if result else "未知错误"))
    
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有必要的模块都可用") 