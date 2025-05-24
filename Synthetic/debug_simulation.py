import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import sys

# 强制使用CPU，避免CUDA错误干扰调试
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 为Jupyter笔记本环境添加一个简单的适配器函数
def jupyter_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()  # 确保输出立即显示

# 添加详细调试输出
DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        jupyter_print("[DEBUG]", *args, **kwargs)

try:
    debug_print("尝试导入模块...")
    from CanonicalComponentCCA import CanonicalComponent
    from WitnessFunction import WitnessFunction
    from Generate_synthetic import GenerateData
    from hsicv2 import hsic_gam_torch as hsic
    
    jupyter_print("所有模块导入成功")
    
    # 一些基本设置
    opt = {
        "n_epochs": 2,          # 减少训练轮数进行快速测试
        "batch_size": 50,       # 使用较小的批量进行调试
        "alpha": 0.009,
        "beta": 8e-5,
        "latent_dim": 1,
        "n_critic": 1,
        "lsmooth": 1
    }
    opt = SimpleNamespace(**opt)
    
    # 使用更小的数据集进行调试
    data_dimensions_debug = {
        "N": 10**3,    # 减少样本数量
        "D": 2,        # 共享组件向量长度
        "D1": 1,       # 私有组件1向量长度 
        "D2": 1,       # 私有组件2向量长度
        "M1": 3,       # 视图1的混合矩阵行数
        "M2": 3,       # 视图2的混合矩阵行数
        "n_epochs": 2  # 减少训练轮数
    }
    
    signal_property_debug = {"mean": 0, "std": 1, "dist": "normal"}
    noise_property_debug = {"mean1": 0, "std1": 1, "mean2": 0, "std2": 1, "dist1": "normal", "dist2": "normal"}
    
    def debug_model_shapes(model, input_tensor, name="模型"):
        """调试函数，打印模型输入和输出的形状"""
        try:
            debug_print(f"{name} 输入形状: {input_tensor.shape}")
            output = model(input_tensor)
            debug_print(f"{name} 输出形状: {output.shape}")
            return output
        except Exception as e:
            debug_print(f"{name} 调用出错: {e}")
            raise

    def debug_simulation(signal_property, noise_property, data_dimensions, anchor_nums=0, random_seed=42):
        jupyter_print("开始调试模拟...")
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        debug_print(f"随机种子已设置为 {random_seed}")
        
        # 确保使用CPU
        Tensor = torch.FloatTensor
        debug_print("使用CPU运行")
        
        # 初始化数据生成器
        data_generate = GenerateData(data_dimensions, opt.batch_size)
        debug_print(f"数据生成器已初始化，批量大小: {opt.batch_size}")
        
        try:
            # 生成数据
            jupyter_print("生成合成数据...")
            mixer_property = {"mean": 0, "std": 1, "dist": "normal"}
            X_1, X_2, S, scir, C1, C2 = data_generate.generate_cca_data(
                signal_property, 
                noise_property, 
                mixer_property, 
                normalize_mean=True, 
                diff_mixture=True
            )
            
            # 打印数据的形状，帮助调试
            debug_print(f"X_1 形状: {X_1.shape}")
            debug_print(f"X_2 形状: {X_2.shape}")
            debug_print(f"S 形状: {S.shape}")
            debug_print(f"C1 形状: {C1.shape if C1 is not None else 'None'}")
            debug_print(f"C2 形状: {C2.shape if C2 is not None else 'None'}")
            
            # 获取混合矩阵
            A1, A2 = data_generate.get_mixing_matrices()
            A1 = A1.type(Tensor)
            A2 = A2.type(Tensor)
            debug_print(f"A1 形状: {A1.shape}")
            debug_print(f"A2 形状: {A2.shape}")
            
            # 将数据转换为变量
            X_1 = Variable(X_1.type(Tensor))
            X_2 = Variable(X_2.type(Tensor))
            S = Variable(S.type(Tensor))
            
            # 处理锚点
            if anchor_nums > 0:
                jupyter_print(f"使用 {anchor_nums} 个锚点")
                random_indices = np.random.choice(X_1.shape[1], size=anchor_nums, replace=False)
                debug_print(f"随机选择的索引: {random_indices}")
                anchors1 = X_1[:, random_indices]
                anchors2 = X_2[:, random_indices]
                debug_print(f"anchors1 形状: {anchors1.shape}")
                debug_print(f"anchors2 形状: {anchors2.shape}")
            else:
                anc_loss = torch.tensor([0.0]).type(Tensor)
                anchors1 = []
                anchors2 = []
            
            # 创建数据加载器
            view1, view2 = data_generate.create_dataloader(X_1, X_2)
            jupyter_print("数据加载器已创建")
            
            # 模型初始化
            jupyter_print("初始化模型...")
            debug_print(f"CanonicalComponent 参数: X_1:{X_1.shape}, D:{data_dimensions['D']}, M1:{data_dimensions['M1']}, N:{data_dimensions['N']}")
            
            try:
                z1 = CanonicalComponent(X_1, data_dimensions["D"], data_dimensions["M1"], data_dimensions["N"], Tensor)
                z2 = CanonicalComponent(X_2, data_dimensions["D"], data_dimensions["M2"], data_dimensions["N"], Tensor)
                f = WitnessFunction(data_dimensions["D"], opt.latent_dim)
                
                z1_p = CanonicalComponent(X_1, data_dimensions["D1"], data_dimensions["M1"], data_dimensions["N"], Tensor)
                z2_p = CanonicalComponent(X_2, data_dimensions["D2"], data_dimensions["M2"], data_dimensions["N"], Tensor)
                jupyter_print("模型初始化成功")
            except Exception as e:
                jupyter_print(f"模型初始化失败: {e}")
                raise
            
            # 优化器
            optimizer_z1 = torch.optim.Adam(z1.parameters(), lr=opt.alpha)
            optimizer_z2 = torch.optim.Adam(z2.parameters(), lr=opt.alpha)
            optimizer_f = torch.optim.Adam(f.parameters(), lr=opt.beta)
            
            optimizer_z1_p = torch.optim.Adam(z1_p.parameters(), lr=1e-3)
            optimizer_z2_p = torch.optim.Adam(z2_p.parameters(), lr=1e-3)
            jupyter_print("优化器已初始化")
            
            # 损失函数和超参数
            lambdaa = 0.1
            lambda_anc = 0.01
            lambda_hsic = 50.0
            loss_func = nn.BCELoss()
            
            # 训练循环
            jupyter_print("开始训练循环...")
            for epoch in range(1, data_dimensions["n_epochs"] + 1):
                try:
                    z_loss1 = 0
                    z_loss2 = 0
                    f_loss = 0
                    
                    jupyter_print(f"Epoch {epoch} - 开始迭代数据加载器")
                    for i, (x1, x2) in enumerate(zip(view1, view2)):
                        debug_print(f"批次 {i+1} - x1 形状: {x1.shape}, x2 形状: {x2.shape}")
                        
                        # 创建标签
                        valid_labels = Variable(Tensor(x1.size(0), 1).fill_(1.0), requires_grad=False)
                        fake_labels = Variable(Tensor(x1.size(0), 1).fill_(0.0), requires_grad=False)
                        
                        # 打乱样本
                        perm = torch.randperm(x1.size(0))
                        x2_shuffle = x2[perm]
                        
                        # 训练判别器
                        jupyter_print("训练判别器...")
                        optimizer_f.zero_grad()
                        
                        # 调试模型输入输出形状
                        try:
                            debug_print("判别器迭代 - 调试形状:")
                            s_x = debug_model_shapes(z1, x1, "z1")
                            s_y = debug_model_shapes(z2, x2, "z2")
                            
                            s_1 = torch.cat([s_x, s_y], dim=1)
                            debug_print(f"s_1 形状 (连接后): {s_1.shape}")
                            
                            s_y_shuffle = debug_model_shapes(z2, x2_shuffle, "z2 (打乱)")
                            s_2 = torch.cat([s_x, s_y_shuffle], dim=1)
                            debug_print(f"s_2 形状 (连接后): {s_2.shape}")
                            
                            f_output1 = debug_model_shapes(f, s_1, "判别器 (真)")
                            f_output2 = debug_model_shapes(f, s_2, "判别器 (假)")
                            
                            # 判别器损失
                            loss_f = loss_func(f_output1, valid_labels) + loss_func(f_output2, fake_labels)
                            debug_print(f"判别器损失: {loss_f.item()}")
                            
                            # 反向传播
                            loss_f.backward(retain_graph=True)
                            optimizer_f.step()
                            f_loss += loss_f.item()
                            
                            jupyter_print("判别器训练完成")
                        except Exception as e:
                            jupyter_print(f"判别器训练失败: {e}")
                            raise
                        
                        # 训练编码器
                        jupyter_print("训练编码器...")
                        try:
                            # 训练 view1 编码器
                            optimizer_z1.zero_grad()
                            optimizer_z1_p.zero_grad()
                            
                            # 锚点损失
                            anc_l = torch.tensor([0.0]).type(Tensor)
                            if anchor_nums > 0:
                                jupyter_print("计算锚点损失...")
                                for j in range(anchor_nums):
                                    debug_print(f"锚点 {j+1} 处理")
                                    anchor1 = anchors1[:, j].reshape(-1, 1)
                                    anchor2 = anchors2[:, j].reshape(-1, 1)
                                    debug_print(f"anchor1 形状: {anchor1.shape}, anchor2 形状: {anchor2.shape}")
                                    
                                    try:
                                        h1 = debug_model_shapes(z1, anchor1, f"z1 (锚点 {j+1})")
                                        h2 = debug_model_shapes(z2, anchor2, f"z2 (锚点 {j+1})")
                                        anc_l += torch.sum((h1 - h2) ** 2)
                                    except Exception as e:
                                        jupyter_print(f"锚点 {j+1} 处理失败: {e}")
                                        raise
                                
                                anc_l = anc_l / anchor_nums
                                debug_print(f"平均锚点损失: {anc_l.item()}")
                            
                            # HSIC损失
                            s_x = z1(x1)
                            p_x = z1_p(x1)
                            debug_print(f"HSIC 计算 - s_x 形状: {s_x.shape}, p_x 形状: {p_x.shape}")
                            
                            hsic_shared = hsic(s_x.detach().reshape(-1, data_dimensions["D"]), 
                                              p_x.reshape(-1, data_dimensions["D1"]))
                            debug_print(f"HSIC 损失: {hsic_shared}")
                            
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
                            
                            hsic_shared = hsic(z2(x2).detach().reshape(-1, data_dimensions["D"]), 
                                              z2_p(x2).reshape(-1, data_dimensions["D2"]))
                            
                            loss_z2 = loss_func(f(s_1), valid_labels) + lambdaa * loss_func(f(s_2), fake_labels) + lambda_hsic * hsic_shared
                            z_loss2 += loss_z2.item()
                            
                            loss_z2.backward(retain_graph=True)
                            optimizer_z2.step()
                            optimizer_z2_p.step()
                            
                            jupyter_print("编码器训练完成")
                        except Exception as e:
                            jupyter_print(f"编码器训练失败: {e}")
                            raise
                    
                    # 输出进度
                    avg_z1 = z_loss1 / len(view1)
                    avg_z2 = z_loss2 / len(view1)
                    avg_f = f_loss / len(view1)
                    jupyter_print(f"[Epoch {epoch}/{data_dimensions['n_epochs']}] [z1: {avg_z1:.6f}] [z2: {avg_z2:.6f}] [f: {avg_f:.6f}] [anc: {anc_l.item():.6f}]")
                    
                except Exception as e:
                    jupyter_print(f"Epoch {epoch} 训练失败: {e}")
                    raise
            
            jupyter_print("训练完成!")
            return {"success": True, "message": "调试模拟成功完成"}
            
        except Exception as e:
            jupyter_print(f"调试模拟失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": str(e)}
    
    # 运行调试模拟
    jupyter_print("开始调试运行...")
    result = debug_simulation(
        signal_property_debug,
        noise_property_debug,
        data_dimensions_debug,
        anchor_nums=3,  # 使用3个锚点
        random_seed=42
    )
    
    # 总结结果
    if result and result["success"]:
        jupyter_print("调试成功: " + result["message"])
    else:
        jupyter_print("调试失败: " + (result["message"] if result else "未知错误"))
    
except ImportError as e:
    jupyter_print(f"模块导入失败: {e}")
    jupyter_print("请确保所有必要的模块都可用") 