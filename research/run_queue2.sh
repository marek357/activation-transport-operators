#!/bin/sh
# Queue 2: finish sweep, then feature identification, then exp1b.
cd "$(dirname "$0")"
while kill -0 5299 2>/dev/null; do sleep 10; done
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk --h 8192 --topk 64 --epochs 4 > results/exp3_topk8k_L5k5.log 2>&1
python3 exp5_feature_transport.py --L 9 --k 1 > results/exp5_L9k1.log 2>&1
python3 exp5_feature_transport.py --L 5 --k 5 --corr topk_h2048_k64 > results/exp5_L5k5.log 2>&1
python3 exp1b_featsel.py 5,5 15,5 11,9 1,9 9,1 19,1 > results/exp1b.log 2>&1
