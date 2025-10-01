# python experiments/predict.py \
#     inference.task=csp \
#     inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt \ # default to null
#     inference.num_samples=1 \   # default to 1
#     inference.num_devices=1 \  
#     inference.sampler.num_timesteps=50


PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict_seq.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/last.ckpt \
    inference.total_samples=5

# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict.py \
#     inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/last.ckpt \
#     inference.num_samples=10 \
#     data.test_sample_limit=10