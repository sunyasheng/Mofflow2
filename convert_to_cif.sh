#!/bin/bash

# 转换预测结果为CIF格式用于VESTA可视化

echo "Converting .pt files to CIF format for VESTA visualization..."

# 设置路径
PREDICTION_PATH="/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/inference/predictions_*.pt"
# PREDICTION_PATH="/ibex/user/suny0a/Proj/MOFFlow-2/logs/gen/inference/predictions_*.pt"
PYTHONPATH="/ibex/user/suny0a/Proj/MOFFlow-2"

# 转换为CIF
PYTHONPATH=$PYTHONPATH python io/pt_to_cif.py \
    --save_pt "$PREDICTION_PATH" \
    --num_samples 1 \
    --num_cpus 4

echo "Conversion completed!"
echo "CIF files saved to: /ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/inference/cif/"
echo ""
echo "To view in VESTA:"
echo "1. Download the CIF files from the server"
echo "2. Open VESTA"
echo "3. File -> Open -> Select the .cif files"
echo ""
echo "CIF files location:"
find /ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/inference/cif/ -name "*.cif" 2>/dev/null || echo "CIF files will be created after running this script"
