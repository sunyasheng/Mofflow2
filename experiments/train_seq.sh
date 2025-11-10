# PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train_seq.py \
#   experiment.name=seq_1step \
#   2>&1 | tee output_seq.log

cd  /ibex/user/suny0a/Proj/MOFFlow-2
source .env_ibex
PYTHONPATH=/ibex/user/suny0a/Proj/MOFFlow-2 python experiments/train_seq.py \
  experiment.name=seq_1step_conditional model.conditional=true \
  2>&1 | tee output_seq_$(date +%Y%m%d_%H%M%S).log
