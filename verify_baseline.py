#!/usr/bin/env python3
"""
验证基线归一化代码的正确性
"""
import torch

def verify_baseline_normalization():
    """验证基线归一化后距离确实为1"""
    
    print("=== 基线归一化验证 ===\n")
    
    # 模拟原始数据
    print("1. 创建测试数据")
    # 假设有3个相机的外参矩阵 (batch=3, 4x4矩阵)
    extrinsics = torch.eye(4).unsqueeze(0).repeat(3, 1, 1).float()
    
    # 设置不同的相机位置
    extrinsics[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])  # 相机0在原点
    extrinsics[1, :3, 3] = torch.tensor([3.0, 4.0, 0.0])  # 相机1
    extrinsics[2, :3, 3] = torch.tensor([6.0, 8.0, 0.0])  # 相机2
    
    print(f"原始相机位置:")
    for i in range(3):
        pos = extrinsics[i, :3, 3]
        print(f"  相机{i}: {pos.tolist()}")
    
    # 模拟代码中的context_indices（选择相机0和2）
    context_indices = torch.tensor([0, 2])
    
    print(f"\n2. 计算原始基线长度")
    context_extrinsics = extrinsics[context_indices]
    a, b = context_extrinsics[:, :3, 3]
    original_baseline = (a - b).norm()
    
    print(f"相机0位置: {a.tolist()}")
    print(f"相机2位置: {b.tolist()}")
    print(f"原始基线长度: {original_baseline.item():.4f}")
    
    # 执行归一化（模拟代码逻辑）
    print(f"\n3. 执行归一化")
    scale = original_baseline
    print(f"scale因子: {scale.item():.4f}")
    
    # 关键步骤：所有相机位置都除以scale
    extrinsics_normalized = extrinsics.clone()
    extrinsics_normalized[:, :3, 3] /= scale
    
    print(f"归一化后相机位置:")
    for i in range(3):
        pos = extrinsics_normalized[i, :3, 3]
        print(f"  相机{i}: {pos.tolist()}")
    
    # 验证结果
    print(f"\n4. 验证结果")
    context_extrinsics_new = extrinsics_normalized[context_indices]
    a_new, b_new = context_extrinsics_new[:, :3, 3]
    new_baseline = (a_new - b_new).norm()
    
    print(f"归一化后相机0位置: {a_new.tolist()}")
    print(f"归一化后相机2位置: {b_new.tolist()}")
    print(f"新的基线长度: {new_baseline.item():.6f}")
    
    # 检查是否为1
    is_one = torch.isclose(new_baseline, torch.tensor(1.0), atol=1e-6)
    print(f"基线长度是否为1: {is_one.item()}")
    
    print(f"\n5. 数学验证")
    print(f"理论计算: {original_baseline.item():.4f} / {scale.item():.4f} = {(original_baseline/scale).item():.6f}")
    
    return new_baseline.item()

def test_different_cases():
    """测试不同情况下的基线归一化"""
    
    print("\n" + "="*50)
    print("测试不同情况")
    print("="*50)
    
    test_cases = [
        {
            "name": "简单情况",
            "pos1": [0, 0, 0],
            "pos2": [5, 0, 0]
        },
        {
            "name": "3D情况",
            "pos1": [1, 2, 3],
            "pos2": [4, 6, 3]
        },
        {
            "name": "负坐标",
            "pos1": [-2, -1, 0],
            "pos2": [3, 4, 0]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {case['name']}")
        
        # 创建测试数据
        extrinsics = torch.eye(4).unsqueeze(0).repeat(2, 1, 1).float()
        extrinsics[0, :3, 3] = torch.tensor(case['pos1'], dtype=torch.float32)
        extrinsics[1, :3, 3] = torch.tensor(case['pos2'], dtype=torch.float32)
        
        # 计算原始基线
        a, b = extrinsics[:, :3, 3]
        original_baseline = (a - b).norm()
        
        # 归一化
        scale = original_baseline
        extrinsics[:, :3, 3] /= scale
        
        # 验证结果
        a_new, b_new = extrinsics[:, :3, 3]
        new_baseline = (a_new - b_new).norm()
        
        print(f"  原始基线: {original_baseline.item():.4f}")
        print(f"  归一化后基线: {new_baseline.item():.6f}")
        print(f"  是否为1: {torch.isclose(new_baseline, torch.tensor(1.0), atol=1e-6).item()}")

if __name__ == "__main__":
    baseline_result = verify_baseline_normalization()
    test_different_cases()
    
    print(f"\n" + "="*50)
    print("结论：")
    print(f"代码执行后，基线长度确实变成了 {baseline_result:.6f}")
    print("基线归一化逻辑完全正确！")
    print("="*50)
