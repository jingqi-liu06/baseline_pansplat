#!/usr/bin/env python3
"""
演示齐次缩放如何保持几何关系不变
"""
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_geometric_preservation():
    """演示几何关系的保持"""
    
    print("=== 几何关系保持不变的演示 ===\n")
    
    # 1. 创建一个简单的3D场景
    print("1. 原始场景设置")
    
    # 三个相机位置
    cameras = np.array([
        [0, 0, 0],      # 相机A（原点）
        [3, 4, 0],      # 相机B
        [6, 0, 2]       # 相机C
    ])
    
    # 一些场景中的物体
    objects = np.array([
        [1.5, 2, 1],    # 物体1
        [4.5, 1, 0.5],  # 物体2
        [2, 3, 1.5]     # 物体3
    ])
    
    print("原始相机位置:")
    for i, cam in enumerate(cameras):
        print(f"  相机{chr(65+i)}: {cam}")
    
    print("原始物体位置:")
    for i, obj in enumerate(objects):
        print(f"  物体{i+1}: {obj}")
    
    # 2. 计算原始的几何关系
    print(f"\n2. 原始几何关系")
    
    # 相机间距离
    dist_AB = np.linalg.norm(cameras[0] - cameras[1])
    dist_AC = np.linalg.norm(cameras[0] - cameras[2])
    dist_BC = np.linalg.norm(cameras[1] - cameras[2])
    
    print(f"相机间距离:")
    print(f"  A-B: {dist_AB:.4f}")
    print(f"  A-C: {dist_AC:.4f}")
    print(f"  B-C: {dist_BC:.4f}")
    
    # 相机到物体的距离
    cam_to_obj = []
    for i, cam in enumerate(cameras):
        for j, obj in enumerate(objects):
            dist = np.linalg.norm(cam - obj)
            cam_to_obj.append(dist)
            print(f"  相机{chr(65+i)}到物体{j+1}: {dist:.4f}")
    
    # 物体间距离
    obj_distances = []
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            dist = np.linalg.norm(objects[i] - objects[j])
            obj_distances.append(dist)
            print(f"  物体{i+1}到物体{j+1}: {dist:.4f}")
    
    # 计算一些角度关系
    print(f"\n角度关系:")
    # 以相机A为顶点，AB和AC的夹角
    vec_AB = cameras[1] - cameras[0]
    vec_AC = cameras[2] - cameras[0]
    angle_BAC = np.arccos(np.dot(vec_AB, vec_AC) / (np.linalg.norm(vec_AB) * np.linalg.norm(vec_AC)))
    angle_BAC_deg = np.degrees(angle_BAC)
    print(f"  ∠BAC: {angle_BAC_deg:.2f}°")
    
    # 3. 执行基线归一化（以A-B作为基线）
    print(f"\n3. 执行基线归一化")
    baseline = dist_AB  # 使用A-B距离作为基线
    scale_factor = 1.0 / baseline
    
    print(f"基线长度: {baseline:.4f}")
    print(f"缩放因子: {scale_factor:.4f}")
    
    # 缩放所有坐标
    cameras_scaled = cameras * scale_factor
    objects_scaled = objects * scale_factor
    
    print("缩放后相机位置:")
    for i, cam in enumerate(cameras_scaled):
        print(f"  相机{chr(65+i)}: {cam}")
    
    print("缩放后物体位置:")
    for i, obj in enumerate(objects_scaled):
        print(f"  物体{i+1}: {obj}")
    
    # 4. 验证几何关系是否保持
    print(f"\n4. 验证几何关系保持不变")
    
    # 重新计算相机间距离
    dist_AB_new = np.linalg.norm(cameras_scaled[0] - cameras_scaled[1])
    dist_AC_new = np.linalg.norm(cameras_scaled[0] - cameras_scaled[2])
    dist_BC_new = np.linalg.norm(cameras_scaled[1] - cameras_scaled[2])
    
    print(f"缩放后相机间距离:")
    print(f"  A-B: {dist_AB_new:.6f} (目标: 1.000000)")
    print(f"  A-C: {dist_AC_new:.6f} (比例: {dist_AC_new/dist_AB_new:.6f}, 原始比例: {dist_AC/dist_AB:.6f})")
    print(f"  B-C: {dist_BC_new:.6f} (比例: {dist_BC_new/dist_AB_new:.6f}, 原始比例: {dist_BC/dist_AB:.6f})")
    
    # 验证比例关系
    print(f"\n比例关系验证:")
    print(f"  原始 AC/AB 比例: {dist_AC/dist_AB:.6f}")
    print(f"  缩放后 AC/AB 比例: {dist_AC_new/dist_AB_new:.6f}")
    print(f"  原始 BC/AB 比例: {dist_BC/dist_AB:.6f}")
    print(f"  缩放后 BC/AB 比例: {dist_BC_new/dist_AB_new:.6f}")
    
    # 验证角度关系
    vec_AB_new = cameras_scaled[1] - cameras_scaled[0]
    vec_AC_new = cameras_scaled[2] - cameras_scaled[0]
    angle_BAC_new = np.arccos(np.dot(vec_AB_new, vec_AC_new) / (np.linalg.norm(vec_AB_new) * np.linalg.norm(vec_AC_new)))
    angle_BAC_new_deg = np.degrees(angle_BAC_new)
    
    print(f"\n角度关系验证:")
    print(f"  原始 ∠BAC: {angle_BAC_deg:.6f}°")
    print(f"  缩放后 ∠BAC: {angle_BAC_new_deg:.6f}°")
    print(f"  角度差异: {abs(angle_BAC_deg - angle_BAC_new_deg):.8f}°")
    
    # 5. 验证相机到物体的比例关系
    print(f"\n5. 相机到物体的比例关系")
    
    cam_to_obj_new = []
    for i, cam in enumerate(cameras_scaled):
        for j, obj in enumerate(objects_scaled):
            dist = np.linalg.norm(cam - obj)
            cam_to_obj_new.append(dist)
    
    print("距离缩放验证:")
    for i, (old_dist, new_dist) in enumerate(zip(cam_to_obj, cam_to_obj_new)):
        scale_ratio = old_dist / new_dist
        print(f"  距离{i+1}: {old_dist:.4f} -> {new_dist:.4f}, 缩放比: {scale_ratio:.6f}")
    
    return cameras, objects, cameras_scaled, objects_scaled

def visualize_scaling():
    """可视化缩放效果"""
    print(f"\n6. 关键理解要点")
    print("="*50)
    
    print("✓ 基线归一化是齐次缩放，所有坐标按相同比例缩放")
    print("✓ 所有距离都按相同比例缩放")
    print("✓ 所有角度关系完全保持不变")
    print("✓ 所有长度比例关系完全保持不变")
    print("✓ 场景的整体几何结构完全一致，只是尺度变小/变大")
    print("✓ 这就像用放大镜或缩小镜看同一个场景")
    
    print(f"\n实际应用意义:")
    print("• 不同场景的基线可能差异很大（室内0.1米 vs 户外10米）")
    print("• 归一化后所有场景的基线都是1，便于模型学习")
    print("• 模型学到的是相对几何关系，而不是绝对尺度")
    print("• 提高了模型的泛化能力和数值稳定性")

if __name__ == "__main__":
    cameras, objects, cameras_scaled, objects_scaled = demonstrate_geometric_preservation()
    visualize_scaling()
