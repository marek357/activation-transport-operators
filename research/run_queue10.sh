#!/bin/sh
# Feature matching through the average Jacobian of the learned operator (data-driven J-lens).
cd "$(dirname "$0")"
while kill -0 15548 2>/dev/null; do sleep 10; done
for p in "1 9" "11 9" "5 5" "15 5"; do
  set -- $p
  python3 exp5_feature_transport.py --L $1 --k $2 --corr topk_h8192_k64 > results/exp5c_L$1k$2.log 2>&1
done
