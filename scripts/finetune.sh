#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-facebook/opt-1.3b}"
MODE="${MODE:-ft}"
GPU="${GPU:-0}"
BS="${BS:-16}"
LR="${LR:-1e-5}"
EPOCH="${EPOCH:-5}"
EVAL_STEPS="${EVAL_STEPS:-200}"
TRAIN="${TRAIN:-1000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LOAD_MODE="${LOAD_MODE:-float16}"
SEED="${SEED:-0}"
REPORT_TO="${REPORT_TO:-none}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-True}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"
BASE_BS="${BS}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="regular"

EXTRA_ARGS=""
if [ "${MODE}" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "${MODE}" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE_OUTPUT_DIR}" = "True" ] || [ "${OVERWRITE_OUTPUT_DIR}" = "true" ]; then
    OVERWRITE_FLAG="--overwrite_output_dir"
fi

LOAD_FLAG=""
if [ "${LOAD_MODE}" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "${LOAD_MODE}" = "float16" ]; then
    LOAD_FLAG="--load_float16"
fi

mkdir -p "${LLM_DIR}/result_uv/${TRAINER}"
cd "${LLM_DIR}"

echo "Tasks to run: ${TASK_LIST[*]}"

for TASK in "${TASK_LIST[@]}"; do
    echo "=========================================="
    echo "Starting task: ${TASK}"
    echo "=========================================="

    DEV="${BASE_DEV}"
    BS="${BASE_BS}"
    TASK_ARGS="--train_as_classification"
    case "${TASK}" in
        CB)
            DEV=100
            ;;
        Copa)
            DEV=100
            TASK_ARGS="--train_as_classification False"
            ;;
        MultiRC)
            GA=$(expr "${BS}" / 2)
            BS=2
            echo "Gradient accumulation: ${GA}"
            TASK_ARGS="--gradient_accumulation_steps ${GA}"
            ;;
        ReCoRD)
            GA=$(expr "${BS}" / 2)
            BS=2
            echo "Gradient accumulation: ${GA}"
            TASK_ARGS="--gradient_accumulation_steps ${GA} --train_as_classification False"
            ;;
        DROP)
            GA=$(expr "${BS}" / 1)
            BS=1
            echo "Gradient accumulation: ${GA}"
            TASK_ARGS="--gradient_accumulation_steps ${GA} --train_as_classification False"
            ;;
        SQuAD)
            TASK_ARGS="--train_as_classification False"
            ;;
    esac

    TAG="${TRAINER}-${MODE}-ep${EPOCH}-bs${BS}-lr${LR}-sd${SEED}-${RUN_SUFFIX}"
    OUTPUT_DIR="${LLM_DIR}/result_uv/${TRAINER}/${TASK}-${MODEL_NAME}-${TAG}"

    echo "Model: ${MODEL}"
    echo "Task: ${TASK}"
    echo "GPU(s): ${GPU}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Trainer: ${TRAINER}"
    echo "Epochs: ${EPOCH}"
    echo "Batch size: ${BS}"
    echo "LR: ${LR}"
    echo "Seed: ${SEED}"
    echo "Mode: ${MODE}"
    echo "Load mode: ${LOAD_MODE}"
    echo "Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"
    echo "Extra args: ${EXTRA_ARGS} ${TASK_ARGS}"

    CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
        --model_name "${MODEL}" \
        --task_name "${TASK}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "${TAG}" \
        --train_set_seed "${SEED}" \
        --num_train "${TRAIN}" \
        --num_dev "${DEV}" \
        --num_eval "${EVAL}" \
        --logging_steps 10 \
        --trainer "${TRAINER}" \
        ${LOAD_FLAG} \
        --learning_rate "${LR}" \
        --num_train_epochs "${EPOCH}" \
        --per_device_train_batch_size "${BS}" \
        --eval_strategy steps \
        --eval_steps "${EVAL_STEPS}" \
        --save_strategy no \
        --save_total_limit 1 \
        --report_to "${REPORT_TO}" \
        ${OVERWRITE_FLAG} \
        ${TASK_ARGS} \
        ${EXTRA_ARGS} \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
