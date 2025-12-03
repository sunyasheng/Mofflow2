# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=csp experiment.name=csp > output.log 2>&1
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=csp experiment.name=csp experiment.warm_start=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/csp/ckpt/epoch_4-step_70093-loss_2.2509.ckpt > output.log 2>&1
# 训练的时候只是数据加载不一样 gen，csp所在的文件路径不一样


PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py \
  experiment.task=gen \
  experiment.name=csp_$(date +%Y%m%d_%H%M%S) \
  experiment.warm_start=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-gen/csp_20251129_095557/ckpt/epoch_69-step_650404-loss_1.3757.ckpt \
  2>&1 | tee output_$(date +%Y%m%d_%H%M%S).log


# /ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-gen/csp_20251120_125600/ckpt/epoch_14-step_139364-loss_3.9131.ckpt

# ========== 多 GPU 使用方法 ==========
# 使用多个 GPU：在命令行中添加 experiment.num_devices=N (N 为 GPU 数量)
# 例如：使用 4 个 GPU
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py \
#   experiment.task=gen \
#   experiment.name=csp_$(date +%Y%m%d_%H%M%S) \
#   experiment.num_devices=2 \
#   2>&1 | tee output_$(date +%Y%m%d_%H%M%S).log

# 使用 8 个 GPU 的示例
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py \
#   experiment.task=gen \
#   experiment.name=csp_$(date +%Y%m%d_%H%M%S) \
#   experiment.num_devices=8 \
#   2>&1 | tee output_$(date +%Y%m%d_%H%M%S).log

# 单 GPU 训练（默认，num_devices=1）
# 添加调试输出，不重定向到 log 文件，直接看输出
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=gen experiment.name=csp_$(date +%Y%m%d_%H%M%S) 2>&1 | tee output_$(date +%Y%m%d_%H%M%S).log

# 或者先用调试模式测试（减少 workers，小数据集）
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py \
#   experiment.task=gen \
#   experiment.name=debug_test \
#   experiment.debug=True \
#   experiment.num_devices=1 \
#   data.train_sample_limit=10 \
#   data.val_sample_limit=5 \
#   data.loader.num_workers=0
