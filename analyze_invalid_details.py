#!/usr/bin/env python3
"""
深入分析 invalid 失败的具体检查项
需要重新运行 MOFChecker 来获取详细的失败原因
"""
import os
import sys
import pickle
import warnings
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.lmdb import read_lmdb
from utils.check_mof_validity import check_mof, get_failed_checks
from pymatgen.core import Structure

def analyze_invalid_details(data_dir, task='csp', max_samples=None):
    """
    分析 invalid 失败的具体检查项
    
    Args:
        data_dir: 数据目录
        task: 任务类型 ('csp' 或 'gen')
        max_samples: 最大分析样本数（None 表示全部）
    """
    print("=" * 80)
    print("深入分析 invalid 失败的具体检查项")
    print("=" * 80)
    print()
    print("注意: 这个脚本需要重新运行 MOFChecker，可能需要较长时间")
    print()
    
    base_dir = os.path.join(data_dir, task)
    splits = ['train', 'val', 'test']
    
    # 所有失败检查项的统计
    all_failed_checks = Counter()
    check_combinations = Counter()  # 失败检查项的组合
    
    total_invalid = 0
    total_processed = 0
    
    for split in splits:
        print(f"处理 {split.upper()} split...")
        
        # 读取 matched 数据
        matched_path = os.path.join(base_dir, f'MetalOxo_matched_{split}_3.lmdb')
        if not os.path.exists(matched_path):
            print(f"  跳过: 文件不存在")
            continue
        
        env = read_lmdb(matched_path)
        with env.begin() as txn:
            cursor = txn.cursor()
            items = list(cursor)
            
            if max_samples:
                items = items[:max_samples]
            
            for key_bytes, value in tqdm(items, desc=f"  {split}", leave=False):
                total_processed += 1
                
                try:
                    feats = pickle.loads(value)
                    
                    # 检查是否使用 matched_coords
                    if feats.get('rmsd') and feats['rmsd'][-1] is None:
                        # rmsd_none，跳过
                        continue
                    
                    # 构建结构
                    coords = feats.get('matched_coords', [feats['gt_coords']])[-1]
                    structure = Structure(
                        lattice=feats['cell_1'],
                        species=feats['atom_types'],
                        coords=coords,
                        coords_are_cartesian=True
                    )
                    
                    # 运行 MOFChecker
                    desc, valid = check_mof(structure)
                    
                    if not valid:
                        total_invalid += 1
                        failed_checks = get_failed_checks(desc)
                        
                        # 统计失败的检查项
                        for check in failed_checks:
                            all_failed_checks[check] += 1
                        
                        # 统计失败检查项的组合（排序后作为 key）
                        if failed_checks:
                            combo_key = tuple(sorted(failed_checks))
                            check_combinations[combo_key] += 1
                
                except Exception as e:
                    # exception 情况，跳过
                    continue
        
        env.close()
        print(f"  {split}: 处理了 {len(items)} 条目")
        print()
    
    # 输出结果
    print("=" * 80)
    print("失败检查项统计:")
    print("=" * 80)
    print(f"总处理数: {total_processed:,}")
    print(f"invalid 数: {total_invalid:,}")
    print()
    
    if all_failed_checks:
        print("各检查项失败次数:")
        for check, count in all_failed_checks.most_common():
            percentage = count / total_invalid * 100 if total_invalid > 0 else 0
            print(f"  {check:35s}: {count:6d} 次 ({percentage:5.1f}%)")
        print()
    
    if check_combinations:
        print("=" * 80)
        print("最常见的失败检查项组合 (Top 10):")
        print("=" * 80)
        for combo, count in check_combinations.most_common(10):
            percentage = count / total_invalid * 100 if total_invalid > 0 else 0
            combo_str = ", ".join(combo)
            print(f"  {combo_str:60s}: {count:6d} 次 ({percentage:5.1f}%)")
        print()
    
    # 检查项说明
    print("=" * 80)
    print("检查项说明:")
    print("=" * 80)
    check_descriptions = {
        "has_carbon": "缺少碳原子",
        "has_hydrogen": "缺少氢原子",
        "has_metal": "缺少金属原子",
        "is_porous": "非多孔结构",
        "has_atomic_overlaps": "存在原子重叠",
        "has_overcoordinated_c": "碳原子过度配位",
        "has_overcoordinated_n": "氮原子过度配位",
        "has_overcoordinated_h": "氢原子过度配位",
        "has_undercoordinated_c": "碳原子欠配位",
        "has_undercoordinated_n": "氮原子欠配位",
        "has_undercoordinated_rare_earth": "稀土元素欠配位",
        "has_undercoordinated_alkali_alkaline": "碱/碱土金属欠配位",
        "has_lone_molecule": "存在孤立分子",
        "has_high_charges": "存在异常高电荷",
        "has_suspicicious_terminal_oxo": "存在可疑末端氧",
        "has_geometrically_exposed_metal": "存在几何暴露金属",
    }
    
    for check, desc in check_descriptions.items():
        if check in all_failed_checks:
            count = all_failed_checks[check]
            percentage = count / total_invalid * 100 if total_invalid > 0 else 0
            print(f"  {check:35s}: {desc:30s} ({count:6d} 次, {percentage:5.1f}%)")
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 invalid 失败的具体检查项')
    parser.add_argument('--data-dir', type=str, 
                       default='/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1/lmdb',
                       help='数据目录')
    parser.add_argument('--task', type=str, default='csp', choices=['csp', 'gen'],
                       help='任务类型')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大分析样本数（用于快速测试，None 表示全部）')
    
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")
    
    if args.max_samples:
        print(f"⚠️  仅分析前 {args.max_samples} 个样本（用于快速测试）")
        print()
    
    analyze_invalid_details(args.data_dir, args.task, args.max_samples)
