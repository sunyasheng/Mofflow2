#!/bin/bash
# 单独运行 Step 5: MOFChecker 有效性检查

# 如果存在 .env 文件，先加载它
if [ -f .env ]; then
    source .env
fi

# 设置/覆盖环境变量（确保使用正确的路径）
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
# 根据要处理的数据集修改 DATA_DIR
export DATA_DIR=/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1
# export DATA_DIR=/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-2.1/Subset_78k
# export DATA_DIR=/ibex/project/c2318/material_discovery/clean_data/preprocessed_data/MOF-DB-2.0

# 检查 DATA_DIR 是否设置
if [ -z "$DATA_DIR" ]; then
    echo "错误: 必须设置 DATA_DIR 环境变量"
    echo "请在脚本中设置 DATA_DIR，或通过环境变量设置"
    exit 1
fi

# 验证输入文件是否存在（检查是否有 matched 文件）
TASK_DIR="${DATA_DIR}/lmdb/csp"
if [ ! -d "$TASK_DIR" ]; then
    echo "错误: 找不到任务目录"
    echo "路径: $TASK_DIR"
    exit 1
fi

# 检查是否有 matched 文件
MATCHED_FILES=$(ls "$TASK_DIR"/MetalOxo_matched_*_*.lmdb 2>/dev/null | wc -l)
if [ "$MATCHED_FILES" -eq 0 ]; then
    echo "警告: 找不到 MetalOxo_matched_*.lmdb 文件"
    echo "路径: $TASK_DIR"
    echo "请确保已经完成 Step 4 (MOF matching)"
    exit 1
fi

echo "=========================================="
echo "Step 5: MOFChecker 有效性检查"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "数据目录: $DATA_DIR"
echo "任务目录: $TASK_DIR"
echo "找到 $MATCHED_FILES 个 matched 文件"
echo "=========================================="
echo ""

# 设置日志文件路径
if [ -z "$LOG_FILE" ]; then
    LOG_FILE="${PROJECT_ROOT}/logs/check_validity_$(date +%Y%m%d_%H%M%S).log"
fi

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""
echo "环境变量验证:"
echo "  DATA_DIR=$DATA_DIR"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo ""

# 验证环境变量是否真的被设置（在 Python 中也会检查）
python -c "import os; print(f'Python 环境中的 DATA_DIR: {os.environ.get(\"DATA_DIR\", \"NOT SET\")}')" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "开始运行 check_mof_validity.py"
echo "=========================================="
echo ""

# 运行 check_mof_validity.py，使用 Hydra 命令行参数直接覆盖配置
# 使用 paths.data_dir 参数直接设置数据目录，确保使用正确的路径
python preprocess/check_mof_validity.py paths.data_dir="$DATA_DIR" "$@" 2>&1 | tee -a "$LOG_FILE"
