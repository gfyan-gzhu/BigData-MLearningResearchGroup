#!/usr/bin/env bash
set -euo pipefail

########################################
# Basic settings
########################################
PYTHON_BIN="python"
MAIN_FILE="main.py"
GPU=0

# Two split settings
SPLITS=(edges etypes)

########################################
# Hyperparameter experiments
# Only on AIFB, clients=5
# 5 random seeds
########################################
HP_DATASET="AIFB"
HP_CLIENTS=5
HP_SEEDS=(1000 2000 3000 4000 5000)

GATING_BASES_LIST=(2 4 8 16)
MAX_PATHS_LIST=(4 8 16 24 32)

# Fixed values when sweeping the other hyperparameter
FIXED_GATING_BASES=8
FIXED_MAX_PATHS=16

########################################
# Ablation experiments
# Only on MUTAG/BGS, clients=5/10
# Only 1 random seed
########################################
ABLATION_DATASETS=(MUTAG BGS)
ABLATION_CLIENTS=(5 10)
ABLATION_SEEDS=(1000)
ABLATIONS=(no_mp uniform static no_residual)

# Do not run extra full-model baseline in ablation
RUN_ABLATION_FULL=false

########################################
# Logging
########################################
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

run_one() {
    local desc="$1"
    shift
    echo "[$(timestamp)] START: ${desc}"
    echo "[$(timestamp)] CMD  : $*"
    "$@"
    echo "[$(timestamp)] DONE : ${desc}"
    echo
}

########################################
# 1) Hyperparameter sweep on AIFB, clients=5
#    split = edges / etypes
########################################

echo "=============================================="
echo "Running hyperparameter experiments on AIFB..."
echo "clients = ${HP_CLIENTS}, splits = ${SPLITS[*]}"
echo "seeds   = ${HP_SEEDS[*]}"
echo "=============================================="

# 1.1 Sweep mp_num_gating_bases, fix mp_max_paths
for split in "${SPLITS[@]}"; do
    for mpb in "${GATING_BASES_LIST[@]}"; do
        for seed in "${HP_SEEDS[@]}"; do
            run_one \
            "chaocan | dataset=${HP_DATASET} | split=${split} | clients=${HP_CLIENTS} | mpb=${mpb} | mpp=${FIXED_MAX_PATHS} | seed=${seed}" \
            "${PYTHON_BIN}" "${MAIN_FILE}" \
            -d "${HP_DATASET}" \
            -s "${split}" \
            -f "FedSP-MPG" \
            -c "${HP_CLIENTS}" \
            --random-seed "${seed}" \
            --mp-num-gating-bases "${mpb}" \
            --mp-max-paths "${FIXED_MAX_PATHS}" \
            --exp-folder "chaocan" \
            -g "${GPU}" \
            2>&1 | tee "${LOG_DIR}/chaocan_${HP_DATASET}_${split}_c${HP_CLIENTS}_mpb${mpb}_mpp${FIXED_MAX_PATHS}_seed${seed}.log"
        done
    done
done

# 1.2 Sweep mp_max_paths, fix mp_num_gating_bases
for split in "${SPLITS[@]}"; do
    for mpp in "${MAX_PATHS_LIST[@]}"; do
        for seed in "${HP_SEEDS[@]}"; do
            run_one \
            "chaocan | dataset=${HP_DATASET} | split=${split} | clients=${HP_CLIENTS} | mpb=${FIXED_GATING_BASES} | mpp=${mpp} | seed=${seed}" \
            "${PYTHON_BIN}" "${MAIN_FILE}" \
            -d "${HP_DATASET}" \
            -s "${split}" \
            -f "FedSP-MPG" \
            -c "${HP_CLIENTS}" \
            --random-seed "${seed}" \
            --mp-num-gating-bases "${FIXED_GATING_BASES}" \
            --mp-max-paths "${mpp}" \
            --exp-folder "chaocan" \
            -g "${GPU}" \
            2>&1 | tee "${LOG_DIR}/chaocan_${HP_DATASET}_${split}_c${HP_CLIENTS}_mpb${FIXED_GATING_BASES}_mpp${mpp}_seed${seed}.log"
        done
    done
done

########################################
# 2) Ablation experiments on MUTAG/BGS
#    clients = 5 / 10
#    split = edges / etypes
#    only 1 seed
########################################

echo "=================================================="
echo "Running ablation experiments on MUTAG and BGS..."
echo "clients = ${ABLATION_CLIENTS[*]}, splits = ${SPLITS[*]}"
echo "seeds   = ${ABLATION_SEEDS[*]}"
echo "=================================================="

# 2.1 Optional full model baseline for ablation comparison
if [ "${RUN_ABLATION_FULL}" = true ]; then
    for dataset in "${ABLATION_DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do
            for clients in "${ABLATION_CLIENTS[@]}"; do
                for seed in "${ABLATION_SEEDS[@]}"; do
                    run_one \
                    "xiaorong-baseline | dataset=${dataset} | split=${split} | clients=${clients} | seed=${seed}" \
                    "${PYTHON_BIN}" "${MAIN_FILE}" \
                    -d "${dataset}" \
                    -s "${split}" \
                    -f "FedSP-MPG" \
                    -c "${clients}" \
                    --random-seed "${seed}" \
                    --mp-num-gating-bases "${FIXED_GATING_BASES}" \
                    --mp-max-paths "${FIXED_MAX_PATHS}" \
                    --exp-folder "xiaorong" \
                    -g "${GPU}" \
                    2>&1 | tee "${LOG_DIR}/xiaorong_baseline_${dataset}_${split}_c${clients}_mpb${FIXED_GATING_BASES}_mpp${FIXED_MAX_PATHS}_seed${seed}.log"
                done
            done
        done
    done
fi

# 2.2 Four ablations
for dataset in "${ABLATION_DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        for clients in "${ABLATION_CLIENTS[@]}"; do
            for abl in "${ABLATIONS[@]}"; do
                for seed in "${ABLATION_SEEDS[@]}"; do
                    run_one \
                    "xiaorong | dataset=${dataset} | split=${split} | clients=${clients} | ablation=${abl} | seed=${seed}" \
                    "${PYTHON_BIN}" "${MAIN_FILE}" \
                    -d "${dataset}" \
                    -s "${split}" \
                    -f "FedSP-MPG" \
                    -a "${abl}" \
                    -c "${clients}" \
                    --random-seed "${seed}" \
                    --mp-num-gating-bases "${FIXED_GATING_BASES}" \
                    --mp-max-paths "${FIXED_MAX_PATHS}" \
                    --exp-folder "xiaorong" \
                    -g "${GPU}" \
                    2>&1 | tee "${LOG_DIR}/xiaorong_${dataset}_${split}_c${clients}_${abl}_mpb${FIXED_GATING_BASES}_mpp${FIXED_MAX_PATHS}_seed${seed}.log"
                done
            done
        done
    done
done

echo "===================================="
echo "All experiments finished successfully."
echo "===================================="