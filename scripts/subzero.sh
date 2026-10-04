#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-facebook/opt-13b}"
MODE="${MODE:-ft}"
GPU="${GPU:-1}"
BS="${BS:-16}"
EPS="${EPS:-1e-3}"
RANK="${RANK:-48}"
TRUNCATION_RANK="${TRUNCATION_RANK:-2}"
ENERGY_THRESHOLD="${ENERGY_THRESHOLD:-0.9}"
STEP_INTERVAL="${STEP_INTERVAL:-1000}"
TRAIN="${TRAIN:-1000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LR="${LR:-1e-7}"
STEPS="${STEPS:-15000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
OPT="${OPT:-sgd}"
REDUCE_RANK="${REDUCE_RANK:-False}"
MULTIPLE_SAMPLE="${MULTIPLE_SAMPLE:-False}"
SWITCH_OPTIM="${SWITCH_OPTIM:-False}"
COMPUTE_VAR="${COMPUTE_VAR:-False}"
MAX_TIME="${MAX_TIME:-0}"
LOAD_MODE="${LOAD_MODE:-float16}"
THRESHOLD="${THRESHOLD:-15000}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="subzero"
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

TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-vi${STEP_INTERVAL}-r${RANK}-mt${MAX_TIME}-${RUN_SUFFIX}"
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
    echo "RANK: ${RANK}"
    echo "STEP INTERVAL: ${STEP_INTERVAL}"
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
        --max_steps "${STEPS}" \
        --trainer "${TRAINER}" \
        ${LOAD_FLAG} \
        --zo_optimizer "${OPT}" \
        --multiple_sample "${MULTIPLE_SAMPLE}" \
        --adam_eps 1e-6 \
        --max_time "${MAX_TIME}" \
        --num_samples "${NUM_SAMPLES}" \
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
        --step_interval "${STEP_INTERVAL}" \
        --rank_r "${RANK}" \
        ${EXTRA_ARGS} \
        ${TASK_ARGS} \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
