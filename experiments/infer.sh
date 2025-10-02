## Generate 3D MOF structures from preds_samples-5.json
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict.py \
    inference.task=gen \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-gen/<your_struct_exp_name>/ckpt/last.ckpt \
    +inference.gen.metal_lib_path=/ibex/user/suny0a/Proj/MOFFlow-2/data/metals/gen/metal_lib_train.pkl \
    +inference.gen.mof_seqs_path=/ibex/user/suny0a/Proj/preds_samples-5.json \
    inference.num_samples=1 \
    inference.sampler.num_timesteps=50
