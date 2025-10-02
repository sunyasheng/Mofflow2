## CSP


PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/epoch_1-step_28038-loss_2.7838.ckpt \
    inference.gen.metal_lib_path=/ibex/user/suny0a/Proj/MOFFlow-2/data/metals/metal_lib_train.pkl \
    inference.gen.mof_seqs_path=/ibex/user/suny0a/Proj/preds_samples-5.json \
    inference.total_samples=1000 \
    inference.temperature=1.0
