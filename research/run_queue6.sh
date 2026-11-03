#!/bin/sh
# After the feature-loss runs (queue3) finish: SAE-code re-scoring on 5->10, then resume queue4.
cd "$(dirname "$0")"
while kill -0 5609 2>/dev/null; do sleep 10; done
python3 exp6_codeeval.py 5 5 gelu_h2048 topk_h2048_k64 topk_h8192_k64 topk_h2048_k64_fl1 topk_h2048_k64_fl3 > results/exp6_L5k5.log 2>&1
kill -CONT 5839
