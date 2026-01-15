#!/usr/bin/env python3
"""
详细分析数据过滤情况，包括 MOFChecker 失败原因
"""
import os
import json
import sys
from pathlib import Path
from collections import Counter

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.lmdb import read_lmdb

def analyze_mofchecker_failures(data_dir, task='csp'):
    """分析 MOFChecker 失败原因"""
    splits = ['train', 'val', 'test']
    
    print("=" * 80)
    print("MOFChecker 失败原因分析")
    print("=" * 80)
    print()
    
    base_dir = os.path.join(data_dir, task)
    total_failures = Counter()
    
    for split in splits:
        json_file = os.path.join(base_dir, f'mofchecker_failure_counts_{split}.json')
        if os.path.exists(json_file):
            print(f"{split.upper()} split:")
            with open(json_file, 'r') as f:
                counts = json.load(f)
            
            for status, count in counts.items():
                print(f"  {status:15s}: {count:6d} 条目")
                total_failures[status] += count
            print()
        else:
            print(f"{split.upper()} split: 文件不存在 ({json_file})")
            print()
    
    if total_failures:
        print("=" * 80)
        print("总计失败原因:")
        print("=" * 80)
        for status, count in total_failures.most_common():
            print(f"  {status:15s}: {count:6d} 条目")
        print()
        
        # 计算失败率
        total_failed = sum(total_failures.values())
        print(f"总失败数: {total_failed:,} 条目")
        
        # 从之前的统计我们知道 Step 3 有 18,355 条目，Step 5 有 9,317 条目
        # 所以失败数应该是 18,355 - 9,317 = 9,038
        print(f"预期失败数: 9,038 条目 (从 Step 3 的 18,355 到 Step 5 的 9,317)")
        print()

def analyze_filtering_details(data_dir, task='csp'):
    """详细分析各步骤的过滤情况"""
    
    print("=" * 80)
    print("详细过滤分析")
    print("=" * 80)
    print()
    
    # 读取各步骤的数据
    splits = ['train', 'val', 'test']
    
    # Step 1: Filtered
    step1_total = 0
    step1_counts = {}
    for split in splits:
        path = os.path.join(data_dir, task, f'MetalOxo_filtered_{split}.lmdb')
        if os.path.exists(path):
            env = read_lmdb(path)
            count = env.stat()['entries']
            step1_counts[split] = count
            step1_total += count
            env.close()
    
    # Step 2: Features
    step2_total = 0
    step2_counts = {}
    for split in splits:
        path = os.path.join(data_dir, task, f'MetalOxo_feats_{split}.lmdb')
        if os.path.exists(path):
            env = read_lmdb(path)
            count = env.stat()['entries']
            step2_counts[split] = count
            step2_total += count
            env.close()
    
    # Step 3: Matched
    step3_total = 0
    step3_counts = {}
    for split in splits:
        path = os.path.join(data_dir, task, f'MetalOxo_matched_{split}_3.lmdb')
        if os.path.exists(path):
            env = read_lmdb(path)
            count = env.stat()['entries']
            step3_counts[split] = count
            step3_total += count
            env.close()
    
    # Step 5: Final
    step5_total = 0
    step5_counts = {}
    for split in splits:
        path = os.path.join(data_dir, task, f'MetalOxo_final_{split}.lmdb')
        if os.path.exists(path):
            env = read_lmdb(path)
            count = env.stat()['entries']
            step5_counts[split] = count
            step5_total += count
            env.close()
    
    # 原始数据
    original_path = os.path.join(data_dir, 'MetalOxo.lmdb')
    original_total = 0
    if os.path.exists(original_path):
        env = read_lmdb(original_path)
        original_total = env.stat()['entries']
        env.close()
    
    print(f"原始数据: {original_total:,} 条目")
    print()
    
    # Step 1 分析
    if step1_total > 0:
        filtered_1 = original_total - step1_total
        print(f"Step 1 (Filter):")
        print(f"  过滤: {filtered_1:,} 条目 ({filtered_1/original_total*100:.1f}%)")
        print(f"  剩余: {step1_total:,} 条目 ({step1_total/original_total*100:.1f}%)")
        print(f"  原因: 大小限制 (BBs>20, 原子>200, CPs>20), 原子重叠, 异常键长")
        print()
    
    # Step 2 分析
    if step2_total > 0 and step1_total > 0:
        filtered_2 = step1_total - step2_total
        print(f"Step 2 (Extract Features):")
        print(f"  过滤: {filtered_2:,} 条目 ({filtered_2/step1_total*100:.1f}%)")
        print(f"  剩余: {step2_total:,} 条目 ({step2_total/original_total*100:.1f}%)")
        print(f"  原因: RDKit 解析失败, 晶格约化失败, 坐标转换错误, 对称性分析失败")
        print()
    
    # Step 3 分析
    if step3_total > 0 and step2_total > 0:
        filtered_3 = step2_total - step3_total
        print(f"Step 3 (MOF Matching):")
        print(f"  过滤: {filtered_3:,} 条目 ({filtered_3/step2_total*100:.1f}%)")
        print(f"  剩余: {step3_total:,} 条目 ({step3_total/original_total*100:.1f}%)")
        print(f"  原因: 匹配失败 (rmsd=None), 结构匹配异常")
        print()
    
    # Step 5 分析
    if step5_total > 0 and step3_total > 0:
        filtered_5 = step3_total - step5_total
        print(f"Step 5 (MOFChecker):")
        print(f"  过滤: {filtered_5:,} 条目 ({filtered_5/step3_total*100:.1f}%) ⚠️ 最大过滤步骤")
        print(f"  剩余: {step5_total:,} 条目 ({step5_total/original_total*100:.1f}%)")
        print(f"  原因: MOFChecker 验证失败, 结构不合理, 连通性问题")
        print()
    
    print("=" * 80)
    print("关键发现:")
    print("=" * 80)
    print(f"1. Step 5 (MOFChecker) 是最大的过滤瓶颈，过滤了 {filtered_5:,} 条目 ({filtered_5/step3_total*100:.1f}%)")
    print(f"2. 最终保留率: {step5_total/original_total*100:.1f}% ({step5_total:,}/{original_total:,})")
    print(f"3. 总过滤率: {(original_total-step5_total)/original_total*100:.1f}% ({original_total-step5_total:,}/{original_total:,})")
    print()

def main():
    default_data_dir = "/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1/lmdb"
    
    data_dir = os.environ.get('DATA_DIR', default_data_dir)
    task = os.environ.get('TASK', 'csp')
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        task = sys.argv[2]
    
    analyze_filtering_details(data_dir, task)
    print()
    analyze_mofchecker_failures(data_dir, task)

if __name__ == "__main__":
    main()
