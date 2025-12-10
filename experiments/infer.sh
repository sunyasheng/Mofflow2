#!/bin/bash

echo "=== MOFFlow-2 完整推理流程 ==="
echo "1. 生成3D MOF结构"
echo "2. 转换为CIF格式用于VESTA可视化"
echo ""

# 设置基础路径变量
PROJECT_ROOT="/ibex/user/suny0a/Proj/MOFFlow-2"

# 检查点选择 (取消注释想要使用的检查点)
CKPT_PATH="$PROJECT_ROOT/logs/mof-gen/csp_20251129_095557/ckpt/epoch_65-step_613243-loss_1.3635.ckpt"  # 自己训练的模型

METAL_LIB_PATH="$PROJECT_ROOT/data/metals/metal_lib_train.pkl"
# METAL_LIB_PATH=/ibex/project/c2318/material_discovery/MOFFLOW2_data/metals/gen/metal_lib_train.pkl

# 定义所有要处理的 target_property 值
TARGET_PROPERTIES=(0.5 0.6 0.7 0.8 0.9 1.0)

# 动态计算基础输出路径 (根据检查点路径)
CKPT_DIR=$(dirname "$CKPT_PATH")           # /logs/mof-gen/csp_xxx/ckpt
PARENT_DIR=$(dirname "$CKPT_DIR")          # /logs/mof-gen/csp_xxx
BASE_INFERENCE_DIR="$PARENT_DIR/inference" # /logs/mof-gen/csp_xxx/inference

echo "=== MOFFlow-2 批量推理流程 ==="
echo "检查点: $CKPT_PATH"
echo "基础输出目录: $BASE_INFERENCE_DIR"
echo "将处理 ${#TARGET_PROPERTIES[@]} 个 target_property: ${TARGET_PROPERTIES[@]}"
echo ""

# 循环处理每个 target_property
for TARGET_PROP in "${TARGET_PROPERTIES[@]}"; do
    echo "=========================================="
    echo "处理 target_property = $TARGET_PROP"
    echo "=========================================="
    
    # 设置序列文件路径
    SEQ_PATH="$PROJECT_ROOT/logs/mof-seq/seq_1step_conditional/inference/temp_1.0_target-${TARGET_PROP}/preds_samples-1000_target-${TARGET_PROP}.json"
    
    # 检查序列文件是否存在
    if [ ! -f "$SEQ_PATH" ]; then
        echo "⚠ 警告: 序列文件不存在，跳过: $SEQ_PATH"
        continue
    fi
    
    # 为每个 target_property 创建独立的输出目录
    INFERENCE_DIR="$BASE_INFERENCE_DIR/target_${TARGET_PROP}"
    mkdir -p "$INFERENCE_DIR"
    
    echo "序列文件: $SEQ_PATH"
    echo "输出目录: $INFERENCE_DIR"
    echo ""
    
    ## Step 1: Generate 3D MOF structures
    echo "步骤1: 生成3D MOF结构 (target_property=$TARGET_PROP)..."
    
    # predict.py 会自动输出到 $PARENT_DIR/inference，所以先清理或移动之前的输出
    DEFAULT_INFERENCE_DIR="$PARENT_DIR/inference"
    if [ -d "$DEFAULT_INFERENCE_DIR" ] && [ "$(ls -A $DEFAULT_INFERENCE_DIR 2>/dev/null)" ]; then
        # 移动之前的输出到临时目录，避免覆盖
        TEMP_DIR="${DEFAULT_INFERENCE_DIR}_temp_$(date +%s)"
        mv "$DEFAULT_INFERENCE_DIR" "$TEMP_DIR" 2>/dev/null || true
    fi
    
    PYTHONPATH="$PROJECT_ROOT" python experiments/predict.py \
        inference.task=gen \
        inference.ckpt_path="$CKPT_PATH" \
        +inference.gen.metal_lib_path="$METAL_LIB_PATH" \
        +inference.gen.mof_seqs_path="$SEQ_PATH" \
        inference.num_samples=1 \
        inference.sampler.num_timesteps=50
    
    # 移动生成的文件到对应的 target_property 目录
    if [ -d "$DEFAULT_INFERENCE_DIR" ]; then
        if [ -f "$DEFAULT_INFERENCE_DIR/predictions_0.pt" ]; then
            mkdir -p "$INFERENCE_DIR"
            mv "$DEFAULT_INFERENCE_DIR/predictions_"*.pt "$INFERENCE_DIR/" 2>/dev/null || true
            # 也移动配置文件
            if [ -f "$DEFAULT_INFERENCE_DIR/config.yaml" ]; then
                mv "$DEFAULT_INFERENCE_DIR/config.yaml" "$INFERENCE_DIR/" 2>/dev/null || true
            fi
        fi
    fi
    
    # 检查第一步是否成功
    if [ -f "$INFERENCE_DIR/predictions_0.pt" ]; then
        echo "✓ 3D结构生成完成"
        echo ""
        
        ## Step 2: Convert to CIF format
        echo "步骤2: 转换为CIF格式 (target_property=$TARGET_PROP)..."
        PREDICTION_PATH="$INFERENCE_DIR/predictions_*.pt"
        
        PYTHONPATH="$PROJECT_ROOT" python io/pt_to_cif.py \
            --save_pt "$PREDICTION_PATH" \
            --num_samples 1 \
            --num_cpus 4
        
        # 检查第二步是否成功
        if [ $? -eq 0 ]; then
            echo "✓ CIF文件转换完成 (target_property=$TARGET_PROP)"
            echo "  CIF文件目录: $INFERENCE_DIR/cif/"
        else
            echo "✗ CIF转换失败 (target_property=$TARGET_PROP)"
        fi
    else
        echo "✗ 3D结构生成失败 (target_property=$TARGET_PROP)"
        echo "  检查序列文件是否存在: $SEQ_PATH"
    fi
    
    echo ""
done

echo "=========================================="
echo "=== 批量处理完成 ==="
echo "=========================================="
echo "所有结果保存在: $BASE_INFERENCE_DIR/"
echo ""
echo "各 target_property 的 CIF 文件位置:"
for TARGET_PROP in "${TARGET_PROPERTIES[@]}"; do
    CIF_DIR="$BASE_INFERENCE_DIR/target_${TARGET_PROP}/cif"
    if [ -d "$CIF_DIR" ]; then
        CIF_COUNT=$(find "$CIF_DIR" -name "*.cif" -type f 2>/dev/null | wc -l)
        echo "  target_${TARGET_PROP}: $CIF_DIR ($CIF_COUNT 个CIF文件)"
    fi
done
echo ""
echo "=== 使用VESTA查看 ==="
echo "1. 下载对应 target_property 目录下的 CIF 文件到本地"
echo "2. 打开VESTA软件"
echo "3. File -> Open -> 选择CIF文件"
echo ""

# 备用检查点路径
# inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/epoch_1-step_28038-loss_2.7838.ckpt \
