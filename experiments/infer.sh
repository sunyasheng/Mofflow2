# python experiments/predict.py \
#     inference.task=csp \
#     inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt \ # default to null
#     inference.num_samples=1 \   # default to 1
#     inference.num_devices=1 \  
#     inference.sampler.num_timesteps=50


PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/predict.py \
    inference.ckpt_path=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/seq/ckpt/last.ckpt \
    inference.num_samples=1 \
    data.test_sample_limit=10 \
    data.loader.batch_size.predict=1 \
    data.loader.num_workers=0
