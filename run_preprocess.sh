#!/bin/bash
# 运行 MOFFlow-2 预处理脚本

# 如果存在 .env 文件，先加载它
if [ -f .env ]; then
    source .env
fi

# 设置/覆盖环境变量（确保使用正确的路径）
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
export DATA_DIR=/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1

# 验证路径是否存在
if [ ! -f "$DATA_DIR/lmdb/MetalOxo.lmdb" ]; then
    echo "错误: 找不到 MetalOxo.lmdb 文件"
    echo "路径: $DATA_DIR/lmdb/MetalOxo.lmdb"
    exit 1
fi

if [ ! -d "$DATA_DIR/splits/csp" ]; then
    echo "错误: 找不到 splits/csp 目录"
    echo "路径: $DATA_DIR/splits/csp"
    exit 1
fi

echo "=========================================="
echo "MOFFlow-2 预处理脚本"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "数据目录: $DATA_DIR"
echo "LMDB 文件: $DATA_DIR/lmdb/MetalOxo.lmdb"
echo "Split 目录: $DATA_DIR/splits/csp"
echo "=========================================="
echo ""
echo "环境变量检查:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  DATA_DIR=$DATA_DIR"
echo ""

# 运行预处理脚本
# 可选参数：
# --mof-matching-repeat N: 指定 MOF matching 步骤重复次数（默认 3）
# --run-conversion: 运行格式转换步骤（默认不运行）
python preprocess.py "$@"
