# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=csp experiment.name=csp > output.log 2>&1
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=csp experiment.name=csp experiment.warm_start=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/csp/ckpt/epoch_4-step_70093-loss_2.2509.ckpt > output.log 2>&1
# 训练的时候只是数据加载不一样 gen，csp所在的文件路径不一样
# 添加调试输出，不重定向到 log 文件，直接看输出
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=gen experiment.name=csp_$(date +%Y%m%d_%H%M%S) 2>&1 | tee output_$(date +%Y%m%d_%H%M%S).log

# 或者先用调试模式测试（减少 workers，小数据集）
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py \
#   experiment.task=gen \
#   experiment.name=debug_test \
#   experiment.debug=True \
#   data.train_sample_limit=10 \
#   data.val_sample_limit=5 \
#   data.loader.num_workers=0
