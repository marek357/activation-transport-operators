#!/bin/sh
cd "$(dirname "$0")"
while kill -0 5450 2>/dev/null; do sleep 10; done
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk --h 2048 --topk 64 --epochs 4 --feat_lambda 1 > results/exp3_topk_fl1_L5k5.log 2>&1
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk --h 2048 --topk 64 --epochs 4 --feat_lambda 3 > results/exp3_topk_fl3_L5k5.log 2>&1
