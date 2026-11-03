#!/bin/sh
# Iteration 3 sweep on 5->10: neuron-basis corrections at ~matched size, then exp1b.
cd "$(dirname "$0")"
python3 exp3_nonlinear.py --L 5 --k 5 --kind geglu   --h 1365 --epochs 4 > results/exp3_geglu_L5k5.log 2>&1
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk    --h 2048 --topk 64 --epochs 4 > results/exp3_topk_L5k5.log 2>&1
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk_ss --h 1024 --topk 64 --epochs 4 > results/exp3_topkss_L5k5.log 2>&1
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk    --h 8192 --topk 64 --epochs 4 > results/exp3_topk8k_L5k5.log 2>&1
python3 exp1b_featsel.py 5,5 15,5 11,9 1,9 9,1 19,1 > results/exp1b.log 2>&1
