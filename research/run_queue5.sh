#!/bin/sh
# Feature transport (with CSLS matching) on all pairs, using the h=8192 top-k corrections.
cd "$(dirname "$0")"
while kill -0 5839 2>/dev/null; do sleep 10; done
for p in "9 1" "5 5" "1 9" "19 1" "15 5" "11 9"; do
  set -- $p
  python3 exp5_feature_transport.py --L $1 --k $2 --corr topk_h8192_k64 > results/exp5b_L$1k$2.log 2>&1
done
