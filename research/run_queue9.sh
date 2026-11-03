#!/bin/sh
# Wide top-k + feature loss lambda=0.3, then code metrics.
cd "$(dirname "$0")"
while kill -0 12061 2>/dev/null; do sleep 10; done
python3 exp3_nonlinear.py --L 5 --k 5 --kind topk --h 8192 --topk 64 --epochs 3 --feat_lambda 0.3 > results/exp3_topk8k_fl0.3_L5k5.log 2>&1
python3 exp6_codeeval.py 5 5 gelu_h2048 topk_h2048_k64 topk_h8192_k64 topk_h2048_k64_fl0.3 topk_h2048_k64_fl1 topk_h2048_k64_fl3 topk_h8192_k64_fl1 topk_h8192_k64_fl0.3 > results/exp6_L5k5.log 2>&1
