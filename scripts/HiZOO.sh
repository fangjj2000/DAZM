#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-facebook/opt-13b}"
MODE="${MODE:-ft}"
GPU="${GPU:-1}"
BS="${BS:-16}"
LR="${LR:-5e-7}"
EPS="${EPS:-1e-3}"
SEED="${SEED:-0}"
TRAIN="${TRAIN:-1000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
STEPS="${STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
WARMUP_STEP="${WARMUP_STEP:-0}"
DECAY_STEP="${DECAY_STEP:-0}"
ZO_LR_SCHEDULER_TYPE="${ZO_LR_SCHEDULER_TYPE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
HESSIAN_SMOOTH_TYPE="${HESSIAN_SMOOTH_TYPE:-constant0}"
LOAD_MODE="${LOAD_MODE:-float16}"
MAX_TIME="${MAX_TIME:-0}"
NO_ALIGN="${NO_ALIGN:-}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="hizoo"
export TRANSFORMERS_OFFLINE=1

EXTRA_ARGS=""
if [ "$MODE" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

LOAD_FLAG=""
if [ "$LOAD_MODE" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" = "float16" ]; then
    LOAD_FLAG="--load_float16"
fi

TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-${HESSIAN_SMOOTH_TYPE}${NO_ALIGN:+-$NO_ALIGN}-${RUN_SUFFIX}"
mkdir -p "${LLM_DIR}/result_uv/${TRAINER}"
cd "${LLM_DIR}"

echo "Tasks to run: ${TASK_LIST[*]}"

for TASK in "${TASK_LIST[@]}"; do
    echo "=========================================="
    echo "Starting task: ${TASK}"
    echo "=========================================="

    DEV="${BASE_DEV}"
    TASK_ARGS="--train_as_classification"
    case "$TASK" in
        CB)
            DEV=100
            ;;
        Copa)
            DEV=100
            TASK_ARGS="--train_as_classification False"
            ;;
        ReCoRD|DROP|SQuAD)
            TASK_ARGS="--train_as_classification False"
            ;;
    esac

    OUTPUT_DIR="${LLM_DIR}/result_uv/${TRAINER}/${TASK}-${MODEL_NAME}-${TAG}"

    echo "TAG: ${TAG}"
    echo "Task: ${TASK}"
    echo "BS: ${BS}"
    echo "LR: ${LR}"
    echo "EPS: ${EPS}"
    echo "SEED: ${SEED}"
    echo "TRAIN/EVAL STEPS: ${STEPS}/${EVAL_STEPS}"
    echo "MODE: ${MODE}"
    echo "Output: ${OUTPUT_DIR}"
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
        --logging_steps 100 \
        --max_steps "${STEPS}" \
        --trainer "${TRAINER}" \
        --max_time "${MAX_TIME}" \
        ${LOAD_FLAG} \
        --learning_rate "${LR}" \
        --zo_eps "${EPS}" \
        --per_device_train_batch_size "${BS}" \
        --lr_scheduler_type constant \
        --load_best_model_at_end \
        --evaluation_strategy steps \
        --save_strategy steps \
        --save_total_limit 1 \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${EVAL_STEPS}" \
        --warmup_step "${WARMUP_STEP}" \
        --decay_step "${DECAY_STEP}" \
        --zo_lr_scheduler_type "${ZO_LR_SCHEDULER_TYPE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --hessian_smooth_type "${HESSIAN_SMOOTH_TYPE}" \
        ${EXTRA_ARGS} \
        ${TASK_ARGS} \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
