#!/bin/bash

echo "=== MOFFlow-2 完整推理流程 ==="
echo "1. 生成3D MOF结构"
echo "2. 转换为CIF格式用于VESTA可视化"
echo ""

# 设置基础路径变量
PROJECT_ROOT="/ibex/user/suny0a/Proj/MOFFlow-2"

# 检查点选择 (取消注释想要使用的检查点)
CKPT_PATH="$PROJECT_ROOT/logs/gen/sp_module/epoch_200-step_414498-loss_0.8644.ckpt"  # 官方训练模型

METAL_LIB_PATH="$PROJECT_ROOT/data/metals/metal_lib_train.pkl"
# SEQ_PATH="$PROJECT_ROOT/logs/mof-seq/seq_1step/inference/temp_1.0_unconditional/preds_samples-250.json"
SEQ_PATH="$PROJECT_ROOT/logs/gen/inference/temp_1.0_unconditional/preds_samples-1000.json" # 官方一阶段训练模型生成的序列

# 删除旧的processed_data.pkl文件，强制重新处理250个序列
PROCESSED_DATA_PATH="$PROJECT_ROOT/logs/mof-seq/seq_1step/inference/temp_1.0_unconditional/processed_data.pkl"
if [ -f "$PROCESSED_DATA_PATH" ]; then
    echo "删除旧的processed_data.pkl文件，强制重新处理250个序列..."
    rm "$PROCESSED_DATA_PATH"
fi

# 动态计算输出路径 (根据检查点路径)
# 从 /logs/gen/sp_module/xxx.ckpt 得到 /logs/gen/inference/
CKPT_DIR=$(dirname "$CKPT_PATH")           # /logs/gen/sp_module
PARENT_DIR=$(dirname "$CKPT_DIR")          # /logs/gen  
INFERENCE_DIR="$PARENT_DIR/inference"      # /logs/gen/inference

echo "=== 路径配置 ==="
echo "检查点: $CKPT_PATH"
echo "输出目录: $INFERENCE_DIR"
echo "序列文件: $SEQ_PATH"
echo ""

## Step 1: Generate 3D MOF structures from preds_samples-5.json
echo "步骤1: 生成3D MOF结构..."
PYTHONPATH="$PROJECT_ROOT" python experiments/predict.py \
    inference.task=gen \
    inference.ckpt_path="$CKPT_PATH" \
    +inference.gen.metal_lib_path="$METAL_LIB_PATH" \
    +inference.gen.mof_seqs_path="$SEQ_PATH" \
    inference.num_samples=10 \
    inference.sampler.num_timesteps=50

# 检查第一步是否成功
if [ $? -eq 0 ]; then
    echo "✓ 3D结构生成完成"
    echo ""
    
    ## Step 2: Convert to CIF format for VESTA visualization
    echo "步骤2: 转换为CIF格式..."
    PREDICTION_PATH="$INFERENCE_DIR/predictions_*.pt"
    
    PYTHONPATH="$PROJECT_ROOT" python io/pt_to_cif.py \
        --save_pt "$PREDICTION_PATH" \
        --num_samples 10 \
        --num_cpus 4
    
    # 检查第二步是否成功
    if [ $? -eq 0 ]; then
        echo "✓ CIF文件转换完成"
        echo ""
        echo "=== 结果文件位置 ==="
        echo "3D结构文件: $INFERENCE_DIR/predictions_0.pt"
        echo "CIF文件目录: $INFERENCE_DIR/cif/"
        echo ""
        echo "=== 使用VESTA查看 ==="
        echo "1. 下载CIF文件到本地"
        echo "2. 打开VESTA软件"
        echo "3. File -> Open -> 选择CIF文件"
        echo ""
        
        # 列出生成的CIF文件
        CIF_DIR="$INFERENCE_DIR/cif"
        if [ -d "$CIF_DIR" ]; then
            echo "生成的CIF文件:"
            find "$CIF_DIR" -name "*.cif" -type f 2>/dev/null | head -10
        fi
    else
        echo "✗ CIF转换失败"
        exit 1
    fi
else
    echo "✗ 3D结构生成失败"
    exit 1
fi

echo ""
echo "=== 流程完成 ==="

# 备用检查点路径
# inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/epoch_1-step_28038-loss_2.7838.ckpt \
