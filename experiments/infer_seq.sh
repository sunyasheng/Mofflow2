
# ### their official released
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 DATA_DIR=/ibex/user/suny0a/Proj/MOFFlow-2/data python experiments/predict_seq.py \
#     inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/gen/seq_module_unconditional/epoch_20-step_10728-loss_0.1081.ckpt \
#     inference.total_samples=1000


# ours unconditional trained with their code
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
#     inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step/ckpt/last.ckpt \
#     inference.total_samples=1000


# ours conditional trained with our code - target property from 0.5 to 0.9
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step_conditional/ckpt/epoch_19-step_45279-loss_0.1291.ckpt \
    inference.total_samples=1000 \
    inference.target_property=0.5

PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step_conditional/ckpt/epoch_19-step_45279-loss_0.1291.ckpt \
    inference.total_samples=1000 \
    inference.target_property=0.6

PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step_conditional/ckpt/epoch_19-step_45279-loss_0.1291.ckpt \
    inference.total_samples=1000 \
    inference.target_property=0.7

PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step_conditional/ckpt/epoch_19-step_45279-loss_0.1291.ckpt \
    inference.total_samples=1000 \
    inference.target_property=0.8

PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step_conditional/ckpt/epoch_19-step_45279-loss_0.1291.ckpt \
    inference.total_samples=1000 \
    inference.target_property=0.9


# logs/mof-seq/seq_1step/inference/temp_1.0_unconditional/preds_samples-5.json
