#!/usr/bin/env python
"""
统计 Step 1 filter 阶段每条被过滤 MOF 的过滤原因。
用法（在 Mofflow2 根目录）：
  export DATA_DIR=/path/to/Subset_78k
  python scripts/analyze_filter_reasons.py [--split train] [--out filtered_reasons.csv]
或使用 hydra 配置：
  python scripts/analyze_filter_reasons.py --config-path=../configs --config-name=base
"""
import os
import sys
import argparse
import pickle
from collections import Counter
from functools import partial

import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

# 项目根目录加入 path，便于 import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mofdiff.common.data_utils import lattice_params_to_matrix_torch, frac_to_cart_coords
from mofdiff.common.atomic_utils import frac2cart, compute_distance_matrix


def get_filter_reason(idx, value, max_bbs=20, max_atoms=200, max_cps=20, prop_list=None):
    """
    返回 (idx, reason_str)。若通过则 reason_str 为 None。
    原因顺序与 preprocess/filter.py 一致，返回第一个失败原因。
    """
    try:
        data = pickle.loads(value)
    except Exception:
        return idx, "pickle_load_error"

    if not prop_list:
        prop_list = []
    try:
        data["y"] = torch.tensor(
            [data.prop_dict[prop] for prop in prop_list], dtype=torch.float32
        ).view(1, -1)
        data["prop_dict"] = {prop: data.prop_dict[prop] for prop in prop_list}
    except Exception:
        return idx, "prop_missing_or_invalid"

    # ----- MOF-level (与 mof_criterion 顺序一致) -----
    try:
        if data.num_components > max_bbs:
            return idx, "num_components>20"
        if data.y.isnan().sum() > 0:
            return idx, "y_has_nan"
        if data.y.isinf().sum() > 0:
            return idx, "y_has_inf"
        cell = lattice_params_to_matrix_torch(data.lengths, data.angles).squeeze()
        distances = compute_distance_matrix(
            cell, frac2cart(data.cg_frac_coords, cell)
        ).fill_diagonal_(5.0)
        if (distances < 1.0).any():
            return idx, "mof_short_distance"
    except Exception:
        return idx, "mof_exception"

    # ----- BB-level (任一 bb 不通过即返回该原因) -----
    for bb in data.bbs:
        try:
            bb.num_cps = bb.is_anchor.long().sum()
            if bb.num_atoms > max_atoms:
                return idx, "bb_num_atoms>200"
            if bb.num_cps > max_cps:
                return idx, "bb_num_cps>20"
            cart_coords = frac_to_cart_coords(
                bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms
            )
            pdist = torch.cdist(cart_coords, cart_coords).fill_diagonal_(5.0)
            edge_index = bb.edge_index
            j, i = edge_index[0], edge_index[1]
            bond_dist = (
                (cart_coords[i] - cart_coords[j]).pow(2).sum(dim=-1).sqrt()
            )
            if pdist.min() <= 0.25:
                return idx, "bb_min_dist<=0.25"
            if bond_dist.max() >= 5.0:
                return idx, "bb_bond_dist>=5.0"
        except Exception:
            return idx, "bb_exception"

    return idx, None  # 通过


def main():
    parser = argparse.ArgumentParser(
        description="统计 Step1 filter 的过滤原因（按原因汇总 + 可选导出每条）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据根目录，默认用环境变量 DATA_DIR",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="csp",
        help="task 名",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="split: train / val / test",
    )
    parser.add_argument(
        "--max-bbs",
        type=int,
        default=20,
        help="与 config preprocess.filter 一致",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=200,
        help="与 config preprocess.filter 一致",
    )
    parser.add_argument(
        "--max-cps",
        type=int,
        default=20,
        help="与 config preprocess.filter 一致",
    )
    parser.add_argument(
        "--prop-list",
        type=str,
        nargs="*",
        default=None,
        help="属性列表，默认用 config 里 CO2 那条；空则 []",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="并行进程数",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="可选：将 (idx, reason) 写入 CSV",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if not data_dir:
        raise SystemExit("ERROR: 请设置 DATA_DIR 或传入 --data-dir")
    data_dir = os.path.expanduser(data_dir)

    # 与 base.yaml 里 csp 的 prop_list 一致（若未传 --prop-list）
    if args.prop_list is None:
        args.prop_list = ["uptake_CO2_0.0004bar_mmolg-1"]

    split_file = os.path.join(
        data_dir, "splits", args.task, f"{args.split}_split.txt"
    )
    lmdb_path = os.path.join(data_dir, "lmdb", "MetalOxo.lmdb")

    if not os.path.isfile(split_file):
        raise SystemExit(f"ERROR: 找不到 split 文件: {split_file}")
    if not os.path.exists(lmdb_path):
        raise SystemExit(f"ERROR: 找不到 LMDB: {lmdb_path}")

    # 读取 split indices
    split_idx = np.loadtxt(split_file, dtype=int)
    if split_idx.ndim == 0:
        split_idx = np.array([int(split_idx)])
    print(f"Split {args.split}: {len(split_idx)} 条")

    # 从 MetalOxo.lmdb 读入 (idx, value)
    import lmdb
    is_file = os.path.isfile(lmdb_path)
    env = lmdb.open(
        lmdb_path, readonly=True, lock=False, subdir=not is_file
    )
    data_items = []
    with env.begin() as txn:
        for idx in tqdm(split_idx, desc="Reading LMDB"):
            key = f"{int(idx)}".encode("ascii")
            value = txn.get(key)
            if value is None:
                continue
            data_items.append((int(idx), value))
    env.close()
    print(f"实际读到: {len(data_items)} 条")

    # 并行统计原因
    fn = partial(
        get_filter_reason,
        max_bbs=args.max_bbs,
        max_atoms=args.max_atoms,
        max_cps=args.max_cps,
        prop_list=args.prop_list,
    )
    results = Parallel(n_jobs=args.num_cpus)(
        delayed(fn)(idx, value) for idx, value in tqdm(data_items, desc="Reasons")
    )

    # 汇总
    reasons = [r for _, r in results if r is not None]
    passed = sum(1 for _, r in results if r is None)
    counter = Counter(reasons)

    print("\n========== 过滤原因统计 ==========")
    print(f"通过: {passed}")
    print(f"被过滤: {len(reasons)}")
    print("\n按原因计数（从多到少）:")
    for reason, count in counter.most_common():
        pct = 100.0 * count / len(reasons) if reasons else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print("==================================\n")

    if args.out:
        with open(args.out, "w") as f:
            f.write("idx,reason\n")
            for idx, reason in results:
                if reason is not None:
                    f.write(f"{idx},{reason}\n")
        print(f"已写入: {args.out}")


if __name__ == "__main__":
    main()
