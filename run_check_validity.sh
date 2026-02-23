#!/bin/bash
# 单独运行 Step 5: MOFChecker 有效性检查

# 如果存在 .env 文件，先加载它
if [ -f .env ]; then
    source .env
fi

# 设置/覆盖环境变量（确保使用正确的路径）
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

# 三个数据目录依次跑
DATA_DIRS=(
    "/home/suny0a/mof_root/material_discovery/clean_data/preprocessed_data/MOF-DB-1.1"
    "/home/suny0a/mof_root/material_discovery/clean_data/preprocessed_data/MOF-DB-2.1/Subset_78k"
    "/home/suny0a/mof_root/material_discovery/clean_data/preprocessed_data/MOF-DB-2.0"
)
# 日志名后缀（与上面顺序对应）
LOG_SUFFIXES=(DB11 DB21 DB20)

for i in "${!DATA_DIRS[@]}"; do
    export DATA_DIR="${DATA_DIRS[$i]}"
    SUFFIX="${LOG_SUFFIXES[$i]}"

    echo ""
    echo "########################################################"
    echo "  [$(($i + 1))/${#DATA_DIRS[@]}] DATA_DIR=$DATA_DIR"
    echo "########################################################"

    # 验证输入文件是否存在（检查是否有 matched 文件）
    TASK_DIR="${DATA_DIR}/lmdb/csp"
    if [ ! -d "$TASK_DIR" ]; then
        echo "错误: 找不到任务目录，跳过"
        echo "路径: $TASK_DIR"
        continue
    fi

    MATCHED_FILES=$(ls "$TASK_DIR"/MetalOxo_matched_*_*.lmdb 2>/dev/null | wc -l)
    if [ "$MATCHED_FILES" -eq 0 ]; then
        echo "警告: 找不到 MetalOxo_matched_*.lmdb 文件，跳过"
        echo "路径: $TASK_DIR"
        continue
    fi

    echo "=========================================="
    echo "Step 5: MOFChecker 有效性检查 ($SUFFIX)"
    echo "=========================================="
    echo "项目根目录: $PROJECT_ROOT"
    echo "数据目录: $DATA_DIR"
    echo "任务目录: $TASK_DIR"
    echo "找到 $MATCHED_FILES 个 matched 文件"
    echo "=========================================="
    echo ""

    # 每个数据集单独日志
    LOG_FILE="${PROJECT_ROOT}/logs/check_validity_${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "日志文件: $LOG_FILE"
    echo "=========================================="
    echo ""

    python -c "import os; print(f'Python 环境中的 DATA_DIR: {os.environ.get(\"DATA_DIR\", \"NOT SET\")}')" 2>&1 | tee -a "$LOG_FILE"
    echo ""
    echo "开始运行 check_mof_validity.py ($SUFFIX) ..."
    echo ""
    python preprocess/check_mof_validity.py paths.data_dir="$DATA_DIR" "$@" 2>&1 | tee -a "$LOG_FILE"
    echo ""
    echo "完成: $SUFFIX"
    echo ""
done

echo "=========================================="
echo "全部 ${#DATA_DIRS[@]} 个数据集已跑完"
echo "=========================================="
