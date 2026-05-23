#!/bin/bash

# ===== 基本设置 =====
PYTHON_BIN=python
SCRIPT=main.py
FRAMEWORK="FedSP-MPG"
MODEL="RGCN"
GPU=0

# ===== 实验组合 =====
#DATASETS=("AIFB" "BGS" "MUTAG")
DATASETS=( "MUTAG")
SPLITS=("edges" "etypes")
CLIENTS=(3 5 10)
SEEDS=(100 450 600 900 1200 1800 2300 3400 3600 4100 4700 5100 5900)

# ===== 日志目录 =====
LOG_ROOT=logs/${FRAMEWORK}
mkdir -p "${LOG_ROOT}"

# ===== 计数 =====
TOTAL=$(( ${#DATASETS[@]} * ${#SPLITS[@]} * ${#CLIENTS[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "总实验数: ${TOTAL}"
echo "开始时间: $(date)"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for client in "${CLIENTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))

        EXP_NAME="${dataset}_${split}_c${client}_seed${seed}"
        LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"

        echo "[${COUNT}/${TOTAL}] 正在运行: ${EXP_NAME}"
        echo "日志文件: ${LOG_FILE}"

        ${PYTHON_BIN} ${SCRIPT} \
          -d "${dataset}" \
          -s "${split}" \
          -f "${FRAMEWORK}" \
          -m "${MODEL}" \
          -c "${client}" \
          -g "${GPU}" \
          --random-seed "${seed}" \
          2>&1 | tee "${LOG_FILE}"

        EXIT_CODE=${PIPESTATUS[0]}

        if [ ${EXIT_CODE} -ne 0 ]; then
          echo "实验失败: ${EXP_NAME} (exit code=${EXIT_CODE})" | tee -a "${LOG_FILE}"
        else
          echo "实验完成: ${EXP_NAME}" | tee -a "${LOG_FILE}"
        fi

        echo "----------------------------------------"
      done
    done
  done
done

echo "========================================"
echo "全部实验结束: $(date)"