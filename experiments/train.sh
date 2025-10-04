# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=csp experiment.name=csp > output.log 2>&1
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train.py experiment.task=csp experiment.name=csp experiment.warm_start=/ibex/user/suny0a/Proj/MOFFlow-2/logs/mof-csp/csp/ckpt/epoch_4-step_70093-loss_2.2509.ckpt > output.log 2>&1
