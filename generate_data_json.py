#!/usr/bin/env python3
"""
生成数据索引JSON文件的示例脚本
使用此脚本可以预先生成JSON文件，后续训练时直接加载JSON而不需要扫描文件夹
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from src.dataset.dataset_mp3d import DatasetMP3D, DatasetMP3DCfg
from src.dataset.view_sampler import ViewSampler
import hydra
from omegaconf import DictConfig

def generate_json_for_dataset():
    """生成数据集的JSON索引文件"""
    
    # 模拟配置
    cfg = DatasetMP3DCfg(
        name="mp3d",
        roots=[Path("/baai-cwm-vepfs/cwm/jingqi.liu/pre_exp/data/pansplat")],
        max_fov=100.0,
        make_baseline_1=False,
        augment=True,
        test_len=-1,
        test_chunk_interval=1,
        test_datasets=[
            {"name": "m3d", "dis": 0.1},
            {"name": "m3d", "dis": 0.25},
            {"name": "m3d", "dis": 0.5},
            {"name": "m3d", "dis": 0.75},
            {"name": "m3d", "dis": 1.0},
            {"name": "residential", "dis": 0.15},
            {"name": "replica", "dis": 0.5},
        ],
        image_shape=[256, 512],
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        skip_bad_shape=True,
        near=0.45,
        far=10.0,
        baseline_scale_bounds=False,
        shuffle_val=False,
        overfit_to_scene=None,
        data_json_path=None  # 设置为None以强制扫描文件夹并生成JSON
    )
    
    # 创建简单的ViewSampler
    class SimpleViewSampler:
        pass
    
    view_sampler = SimpleViewSampler()
    
    # 为不同阶段生成JSON文件
    stages = ["train", "val", "test"]
    
    for stage in stages:
        print(f"Generating JSON for stage: {stage}")
        try:
            dataset = DatasetMP3D(cfg, stage, view_sampler)
            print(f"Generated {len(dataset.data)} data entries for {stage}")
            print(f"JSON file saved for {stage} stage")
        except Exception as e:
            print(f"Error generating JSON for {stage}: {e}")
        print("-" * 50)

if __name__ == "__main__":
    generate_json_for_dataset()
