## Generate 3D MOF structures from preds_samples-5.json
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict.py \
    inference.task=gen \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/epoch_1-step_28038-loss_2.7838.ckpt \
    +inference.gen.metal_lib_path=/ibex/user/suny0a/Proj/MOFFlow-2/data/metals/metal_lib_train.pkl \
    +inference.gen.mof_seqs_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-seq/seq_1step/inference/temp_1.0_unconditional/preds_samples-5.json \
    inference.num_samples=1 \
    inference.sampler.num_timesteps=50
