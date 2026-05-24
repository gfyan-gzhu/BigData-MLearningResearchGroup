#!/bin/bash

# 切到当前脚本所在目录（P2P）
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

SEEDS=(1000 2000 3000 4000 5000)

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    REPEAT=$((i+1))

    echo "============================================================"
    echo "Running repeat $REPEAT / 5 | seed=$SEED"
    echo "============================================================"

    python3 main.py \
        --dataset biochem \
        --num_rounds 200 \
        --local_epoch 1 \
        --seed "$SEED" \
        --repeat "$REPEAT" \
        --device cpu
done