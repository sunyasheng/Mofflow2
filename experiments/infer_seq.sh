# original
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step/ckpt/last.ckpt \
    inference.total_samples=1000

# ours
# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
#     inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step_conditional/ckpt/epoch_19-step_45279-loss_0.1291.ckpt \
#     inference.total_samples=250
#     # inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step/ckpt/last.ckpt \


# logs/mof-seq/seq_1step/inference/temp_1.0_unconditional/preds_samples-5.json
