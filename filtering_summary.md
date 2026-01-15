# MOFFlow-2 数据过滤分析总结

## 📊 实际统计数据

### 数据流程概览
- **原始数据**: 25,992 条目
- **最终数据**: 9,317 条目
- **总过滤率**: 64.2% (16,675 条目被过滤)

### 各步骤过滤详情

| 步骤 | 剩余条目 | 过滤条目 | 过滤率 | 累计保留率 |
|------|---------|---------|--------|-----------|
| **原始** | 25,992 | - | - | 100.0% |
| **Step 1: Filter** | 21,289 | 4,703 | 18.1% | 81.9% |
| **Step 2: Extract Features** | 18,523 | 2,766 | 13.0% | 71.3% |
| **Step 3: MOF Matching** | 18,355 | 168 | 0.9% | 70.6% |
| **Step 5: MOFChecker** | **9,317** | **9,038** | **49.2%** | **35.8%** |

---

## 🔍 关键发现

### 1. Step 5 (MOFChecker) 是最大瓶颈
- **过滤了 9,038 条目 (49.2%)**
- 这是导致最终数据量大幅减少的主要原因

### 2. MOFChecker 失败原因分布

| 失败类型 | 数量 | 占比 | 说明 |
|---------|------|------|------|
| **invalid** | 8,271 | 91.5% | MOFChecker 验证失败 |
| **rmsd_none** | 625 | 6.9% | MOF matching 失败 |
| **exception** | 142 | 1.6% | 处理异常错误 |

---

## 📋 MOFChecker 验证标准

根据 `utils/check_mof_validity.py`，MOFChecker 检查以下条件：

### ✅ 必需条件 (必须为 True)
- `has_carbon`: 必须包含碳原子
- `has_hydrogen`: 必须包含氢原子
- `has_metal`: 必须包含金属原子
- `is_porous`: 必须是多孔结构

### ❌ 禁止条件 (必须为 False)
- `has_atomic_overlaps`: 不能有原子重叠
- `has_overcoordinated_c/n/h`: 不能有过度配位的 C/N/H
- `has_undercoordinated_c/n/rare_earth/alkali_alkaline`: 不能有欠配位原子
- `has_lone_molecule`: 不能有孤立分子片段
- `has_high_charges`: 不能有异常高电荷
- `has_suspicicious_terminal_oxo`: 不能有可疑末端氧
- `has_geometrically_exposed_metal`: 不能有几何暴露的金属

**注意**: `has_3d_connected_graph` 检查在代码中被跳过

---

## 💡 失败原因分析

### invalid (8,271 条目, 91.5%)
这是最主要的失败原因，可能包括：
- 缺少必需元素（无碳、无氢、无金属）
- 原子重叠（原子间距离过近）
- 配位问题（过度配位或欠配位）
- **非多孔结构** (`is_porous = False`) - 可能是主要原因
- 孤立分子片段
- 异常电荷
- 可疑末端氧
- 几何暴露金属

### rmsd_none (625 条目, 6.9%)
MOF matching 步骤失败：
- 匹配坐标生成失败
- 结构匹配算法无法找到有效匹配
- 可能是金属库不完整或匹配参数过于严格

### exception (142 条目, 1.6%)
处理异常：
- 结构构建失败
- MOFChecker 运行时错误
- 数据格式问题

---

## 🛠️ 改进建议

### 1. 针对 invalid 失败 (91.5%)

**选项 A: 放宽验证标准**
- 修改 `utils/check_mof_validity.py` 中的 `EXPECTED_CHECK_VALUES`
- 考虑放宽 `is_porous` 检查（如果非多孔结构也可接受）
- 放宽配位检查的容差

**选项 B: 分析具体失败原因**
- 创建一个脚本，对失败的 MOF 进行详细分析
- 统计哪些检查失败最多（如 `is_porous`、`has_atomic_overlaps` 等）
- 针对性调整验证标准

**选项 C: 预处理修复**
- 在 Step 5 之前添加结构修复步骤
- 修复原子重叠问题
- 修复配位异常

### 2. 针对 rmsd_none 失败 (6.9%)

- 增加 MOF matching 的优化迭代次数（`maxiter`）
- 增加种群大小（`popsize`）
- 调整匹配算法的容差参数（`ltol`, `stol`, `angle_tol`）
- 检查金属库是否完整

### 3. 针对 exception 失败 (1.6%)

- 查看详细错误日志
- 检查异常数据的特征
- 添加异常处理，跳过明显有问题的数据

---

## 📈 数据分布

### 按 Split 分布

| Split | Step 3 (Matched) | Step 5 (Final) | 过滤数 | 过滤率 |
|-------|-----------------|----------------|--------|--------|
| **train** | 14,802 | 7,499 | 7,303 | 49.3% |
| **val** | 1,763 | 940 | 823 | 46.7% |
| **test** | 1,790 | 878 | 912 | 50.9% |

### 按失败类型分布 (Step 5)

| Split | success | invalid | rmsd_none | exception |
|-------|---------|---------|-----------|-----------|
| **train** | 7,499 | 6,654 | 525 | 124 |
| **val** | 940 | 771 | 43 | 9 |
| **test** | 878 | 846 | 57 | 9 |

---

## 🔧 实用工具

### 检查数据统计
```bash
python check_data_counts.py
```

### 详细分析
```bash
python analyze_filtering.py
```

### 查看失败原因
```bash
cat /ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1/lmdb/csp/mofchecker_failure_counts_{split}.json
```

---

## 📝 结论

1. **主要瓶颈**: Step 5 (MOFChecker) 过滤了 49.2% 的数据
2. **主要失败原因**: `invalid` 占 91.5%，其中 `is_porous` 可能是主要原因
3. **最终保留率**: 35.8% (9,317/25,992)
4. **建议**: 优先分析 `invalid` 失败的具体原因，考虑放宽 `is_porous` 等验证标准
