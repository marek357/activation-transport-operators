#!/bin/sh
# Width / sparsity sweep for the sparse correction on 15->20.
cd "$(dirname "$0")"
while kill -0 10711 2>/dev/null; do sleep 10; done
python3 exp3_nonlinear.py --L 15 --k 5 --kind topk --h 16384 --topk 64 --epochs 3 > results/exp3_topk16k_L15k5.log 2>&1
python3 exp3_nonlinear.py --L 15 --k 5 --kind topk --h 8192 --topk 128 --epochs 3 > results/exp3_topk8k_k128_L15k5.log 2>&1
python3 exp3_nonlinear.py --L 15 --k 5 --kind topk --h 8192 --topk 32 --epochs 3 > results/exp3_topk8k_k32_L15k5.log 2>&1
