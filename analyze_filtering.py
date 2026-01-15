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
        
        # 计算失败率（不包括 success）
        total_failed = sum(count for status, count in total_failures.items() if status != 'success')
        total_processed = sum(total_failures.values())
        success_count = total_failures.get('success', 0)
        
        print("=" * 80)
        print("失败统计汇总:")
        print("=" * 80)
        print(f"总处理数: {total_processed:,} 条目")
        print(f"成功数: {success_count:,} 条目")
        print(f"总失败数: {total_failed:,} 条目 ({total_failed/total_processed*100:.1f}%)")
        print()
        
        if total_failed > 0:
            print("失败原因分布:")
            for status, count in sorted(total_failures.items(), key=lambda x: x[1], reverse=True):
                if status != 'success':
                    percentage = count / total_failed * 100
                    print(f"  {status:15s}: {count:6d} 条目 ({percentage:5.1f}%)")
            print()
            
            # 详细说明各失败原因
            print("失败原因说明:")
            if 'invalid' in total_failures:
                print(f"  invalid ({total_failures['invalid']:,} 条目): MOFChecker 验证失败")
                print("    可能原因: 原子重叠、配位异常、孤立分子、异常电荷、")
                print("              非多孔结构、3D连通性问题等")
            if 'rmsd_none' in total_failures:
                print(f"  rmsd_none ({total_failures['rmsd_none']:,} 条目): MOF 匹配失败，无法计算 RMSD")
            if 'exception' in total_failures:
                print(f"  exception ({total_failures['exception']:,} 条目): 处理过程中发生异常")
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
    print("=" * 80)
    print("MOFChecker 验证标准 (来自 utils/check_mof_validity.py):")
    print("=" * 80)
    print("必须满足的条件:")
    print("  ✓ has_carbon: True (必须有碳)")
    print("  ✓ has_hydrogen: True (必须有氢)")
    print("  ✓ has_metal: True (必须有金属)")
    print("  ✓ is_porous: True (必须多孔)")
    print()
    print("必须不满足的条件 (否则标记为 invalid):")
    print("  ✗ has_atomic_overlaps: False (无原子重叠)")
    print("  ✗ has_overcoordinated_c/n/h: False (C/N/H 配位不过度)")
    print("  ✗ has_undercoordinated_c/n/rare_earth/alkali_alkaline: False (配位不不足)")
    print("  ✗ has_lone_molecule: False (无孤立分子)")
    print("  ✗ has_high_charges: False (无异常电荷)")
    print("  ✗ has_suspicicious_terminal_oxo: False (无非正常末端氧)")
    print("  ✗ has_geometrically_exposed_metal: False (无几何暴露金属)")
    print()
    print("注意: has_3d_connected_graph 在验证中被跳过")
    print()

def print_mofchecker_explanation():
    """解释 MOFChecker 失败原因的含义"""
    print("=" * 80)
    print("MOFChecker 失败原因说明")
    print("=" * 80)
    print()
    print("失败类型说明:")
    print()
    print("1. invalid (8,271 条目, 91.5%):")
    print("   MOFChecker 验证失败，可能的原因包括:")
    print("   - 缺少必需元素: 无碳、无氢、无金属")
    print("   - 原子重叠: 原子间距离过近")
    print("   - 配位问题: 过度配位或欠配位的原子")
    print("   - 非多孔结构: is_porous = False")
    print("   - 孤立分子: 存在未连接的分子片段")
    print("   - 高电荷: 存在异常高的原子电荷")
    print("   - 可疑末端氧: 存在不合理的末端氧原子")
    print("   - 几何暴露金属: 金属原子几何位置不合理")
    print()
    print("2. rmsd_none (625 条目, 6.9%):")
    print("   MOF matching 步骤失败，无法计算 RMSD")
    print("   - 匹配坐标生成失败")
    print("   - 结构匹配算法无法找到有效匹配")
    print()
    print("3. exception (142 条目, 1.6%):")
    print("   处理过程中发生异常错误")
    print("   - 结构构建失败")
    print("   - MOFChecker 运行时错误")
    print("   - 数据格式问题")
    print()
    print("=" * 80)
    print("改进建议:")
    print("=" * 80)
    print()
    print("1. 对于 invalid 失败 (91.5%):")
    print("   - 检查 MOFChecker 的验证标准是否过于严格")
    print("   - 考虑放宽某些条件（如 is_porous、配位检查）")
    print("   - 分析哪些检查失败最多，针对性调整")
    print()
    print("2. 对于 rmsd_none 失败 (6.9%):")
    print("   - 增加 MOF matching 的优化迭代次数")
    print("   - 调整匹配算法的容差参数")
    print("   - 检查金属库是否完整")
    print()
    print("3. 对于 exception 失败 (1.6%):")
    print("   - 查看详细错误日志")
    print("   - 检查异常数据的特征")
    print("   - 可能需要预处理修复某些结构问题")
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
    print()
    print_mofchecker_explanation()

if __name__ == "__main__":
    main()
