#!/bin/sh
# After feature transport (queue5): code metrics on all pairs, then feature-loss sweet-spot runs.
cd "$(dirname "$0")"
while kill -0 6402 2>/dev/null; do sleep 10; done
for p in "9 1" "1 9" "19 1" "15 5" "11 9"; do
  set -- $p
  python3 exp6_codeeval.py $1 $2 topk_h8192_k64 > results/exp6_L$1k$2.log 2>&1
done
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk --h 8192 --topk 64 --epochs 3 --feat_lambda 1 > results/exp3_topk8k_fl1_L5k5.log 2>&1
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk --h 2048 --topk 64 --epochs 4 --feat_lambda 0.3 > results/exp3_topk_fl0.3_L5k5.log 2>&1
python3 exp6_codeeval.py 5 5 gelu_h2048 topk_h2048_k64 topk_h8192_k64 topk_h2048_k64_fl0.3 topk_h2048_k64_fl1 topk_h2048_k64_fl3 topk_h8192_k64_fl1 > results/exp6_L5k5.log 2>&1
