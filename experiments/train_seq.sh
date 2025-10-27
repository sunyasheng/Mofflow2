# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train_seq.py \
#   experiment.name=seq_1step \
#   2>&1 | tee output_seq.log

PYTHONPATH=/home/suny0a/ibex_root/Proj/MOFFlow-2 python experiments/train_seq.py \
  experiment.name=seq_1step_conditional model.conditional=true \
  2>&1 | tee output_seq.log
