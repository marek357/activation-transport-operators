#!/bin/sh
cd "$(dirname "$0")"
while kill -0 5609 2>/dev/null; do sleep 10; done
for p in "9 1" "19 1" "15 5" "1 9" "11 9"; do
  set -- $p
  python3 exp3_nonlinear.py --L $1 --k $2 --kind topk --h 8192 --topk 64 --epochs 3 > results/exp3_topk8k_L$1k$2.log 2>&1
done
