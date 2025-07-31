#!/usr/bin/env python3
"""
演示 einops 的 repeat 语法
"""
import torch
from einops import repeat, rearrange

def demonstrate_einops_syntax():
    """演示 einops 的语法"""
    
    print("=== einops 语法演示 ===\n")
    
    # 1. 创建一个简单的2D张量 (相机内参矩阵)
    print("1. 原始张量 (单个内参矩阵)")
    intrinsics = torch.eye(3, dtype=torch.float32)
    print(f"形状: {intrinsics.shape}")
    print(f"内容:\n{intrinsics}")
    
    # 2. 使用 einops 的 repeat 语法
    print(f"\n2. 使用 einops repeat")
    print("代码: repeat(intrinsics, 'h w -> b h w', b=4)")
    
    # 这里 h=3, w=3 (原始矩阵的高和宽)
    # 要转换为 b=4, h=3, w=3 (批量大小为4)
    batch_intrinsics = repeat(intrinsics, "h w -> b h w", b=4)
    
    print(f"新形状: {batch_intrinsics.shape}")
    print(f"含义: 将 (3,3) 扩展为 (4,3,3)")
    print(f"内容:\n{batch_intrinsics}")
    
    # 3. 等价的 PyTorch 原生写法
    print(f"\n3. 等价的 PyTorch 原生写法")
    
    # 方法1: unsqueeze + expand
    batch_intrinsics_v1 = intrinsics.unsqueeze(0).expand(4, -1, -1)
    print(f"方法1 (unsqueeze+expand): {batch_intrinsics_v1.shape}")
    
    # 方法2: repeat (注意这是 torch.repeat，不是 einops.repeat)
    batch_intrinsics_v2 = intrinsics.unsqueeze(0).repeat(4, 1, 1)
    print(f"方法2 (unsqueeze+repeat): {batch_intrinsics_v2.shape}")
    
    # 方法3: 手动复制
    batch_intrinsics_v3 = torch.stack([intrinsics] * 4, dim=0)
    print(f"方法3 (stack): {batch_intrinsics_v3.shape}")
    
    # 验证结果相同
    print(f"\n4. 验证结果是否相同")
    print(f"einops vs 方法1: {torch.equal(batch_intrinsics, batch_intrinsics_v1)}")
    print(f"einops vs 方法2: {torch.equal(batch_intrinsics, batch_intrinsics_v2)}")
    print(f"einops vs 方法3: {torch.equal(batch_intrinsics, batch_intrinsics_v3)}")

def more_einops_examples():
    """更多 einops 例子"""
    
    print(f"\n5. 更多 einops 语法例子")
    print("="*40)
    
    # 创建测试数据
    img = torch.randn(3, 256, 512)  # RGB图像: channels, height, width
    print(f"原始图像形状: {img.shape} (C, H, W)")
    
    # 例子1: 添加批次维度
    batch_img = repeat(img, "c h w -> b c h w", b=8)
    print(f"添加批次维度: {batch_img.shape} (B, C, H, W)")
    
    # 例子2: 重新排列维度
    img_hwc = rearrange(img, "c h w -> h w c")
    print(f"CHW->HWC: {img_hwc.shape} (H, W, C)")
    
    # 例子3: 复杂的重排
    patches = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=16, p2=16)
    print(f"图像分块: {patches.shape} (num_patches, patch_features)")
    
    # 例子4: 在代码中看到的实际用法
    print(f"\n6. 代码中的实际用法")
    print("="*40)
    
    # 模拟代码中的情况
    single_intrinsics = torch.eye(3)
    num_cameras = 3  # 假设有3个相机
    
    print(f"单个内参矩阵: {single_intrinsics.shape}")
    print("代码: repeat(intrinsics, 'h w -> b h w', b=len(extrinsics))")
    
    batch_intrinsics = repeat(single_intrinsics, "h w -> b h w", b=num_cameras)
    print(f"批量内参矩阵: {batch_intrinsics.shape}")
    print("含义: 为每个相机创建相同的内参矩阵")

def why_use_einops():
    """为什么使用 einops"""
    
    print(f"\n7. 为什么使用 einops？")
    print("="*40)
    
    print("✓ 可读性强: 'h w -> b h w' 比 unsqueeze(0).repeat(b,1,1) 更清晰")
    print("✓ 不容易出错: 维度名称让操作意图明确")
    print("✓ 自文档化: 代码即注释，一眼就知道在做什么")
    print("✓ 维度安全: 会检查维度是否匹配")
    print("✓ 支持多种后端: PyTorch, TensorFlow, NumPy, JAX")
    
    print(f"\n传统方式 vs einops:")
    print("传统: tensor.unsqueeze(0).expand(4, -1, -1)")
    print("einops: repeat(tensor, 'h w -> b h w', b=4)")
    print("哪个更容易理解？显然是 einops！")

if __name__ == "__main__":
    demonstrate_einops_syntax()
    more_einops_examples()
    why_use_einops()
