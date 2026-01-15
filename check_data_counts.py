#!/usr/bin/env python3
"""
检查预处理各步骤的数据条目数
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.lmdb import read_lmdb

def check_counts(data_dir, task='csp'):
    """检查各步骤的数据条目数"""
    
    splits = ['train', 'val', 'test']
    steps = [
        ('原始数据', f'MetalOxo.lmdb', None),
        ('Step 1: Filtered', f'{task}/MetalOxo_filtered_{{split}}.lmdb', 'split'),
        ('Step 2: Features', f'{task}/MetalOxo_feats_{{split}}.lmdb', 'split'),
        ('Step 3: Matched (trial 3)', f'{task}/MetalOxo_matched_{{split}}_3.lmdb', 'split'),
        ('Step 5: Final', f'{task}/MetalOxo_final_{{split}}.lmdb', 'split'),
    ]
    
    print("=" * 80)
    print("MOFFlow-2 数据过滤统计")
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"任务: {task}")
    print("=" * 80)
    print()
    
    total_counts = {}
    
    for step_name, pattern, split_type in steps:
        print(f"{step_name}:")
        step_total = 0
        
        if split_type == 'split':
            for split in splits:
                path = os.path.join(data_dir, pattern.format(split=split))
                if os.path.exists(path):
                    try:
                        env = read_lmdb(path)
                        count = env.stat()['entries']
                        print(f"  {split:6s}: {count:6d} 条目")
                        step_total += count
                        env.close()
                    except Exception as e:
                        print(f"  {split:6s}: 错误 - {e}")
                else:
                    print(f"  {split:6s}: 文件不存在")
        else:
            path = os.path.join(data_dir, pattern)
            if os.path.exists(path):
                try:
                    env = read_lmdb(path)
                    count = env.stat()['entries']
                    print(f"  总计: {count:6d} 条目")
                    step_total = count
                    env.close()
                except Exception as e:
                    print(f"  总计: 错误 - {e}")
            else:
                print(f"  总计: 文件不存在")
        
        total_counts[step_name] = step_total
        if step_total > 0:
            print(f"  小计: {step_total:6d} 条目")
        print()
    
    # 计算过滤比例
    print("=" * 80)
    print("过滤统计:")
    print("=" * 80)
    
    if '原始数据' in total_counts and total_counts['原始数据'] > 0:
        original = total_counts['原始数据']
        print(f"原始数据: {original:,} 条目")
        print()
        
        prev_count = original
        for step_name in ['Step 1: Filtered', 'Step 2: Features', 'Step 3: Matched (trial 3)', 'Step 5: Final']:
            if step_name in total_counts and total_counts[step_name] > 0:
                current = total_counts[step_name]
                filtered = prev_count - current
                filter_rate = (filtered / prev_count * 100) if prev_count > 0 else 0
                remaining_rate = (current / original * 100) if original > 0 else 0
                
                print(f"{step_name}:")
                print(f"  剩余: {current:,} 条目 ({remaining_rate:.1f}% 原始数据)")
                print(f"  过滤: {filtered:,} 条目 ({filter_rate:.1f}% 上一步)")
                print()
                
                prev_count = current

if __name__ == "__main__":
    # 默认数据目录
    default_data_dir = "/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1/lmdb"
    
    # 可以从环境变量或命令行参数获取
    data_dir = os.environ.get('DATA_DIR', default_data_dir)
    task = os.environ.get('TASK', 'csp')
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        task = sys.argv[2]
    
    check_counts(data_dir, task)
