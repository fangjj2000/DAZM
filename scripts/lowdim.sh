#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-facebook/opt-13b}"
MODE="${MODE:-ft}"
GPU="${GPU:-1}"
BS="${BS:-16}"
EPS="${EPS:-1e-3}"
LR="${LR:-1e-2}"
STEPS="${STEPS:-6000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
TRAIN="${TRAIN:-1000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LOAD_MODE="${LOAD_MODE:-float16}"
MAX_TIME="${MAX_TIME:-0}"
SEED="${SEED:-0}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-gemma}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-True}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

RANK="${RANK:-64}"
STEP_INTERVAL="${STEP_INTERVAL:-100}"
OPT="${OPT:-muon}"
MULTIPLE_SAMPLE="${MULTIPLE_SAMPLE:-True}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="lowdim"
export TRANSFORMERS_OFFLINE=1

EXTRA_ARGS=""
if [ "$MODE" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE_OUTPUT_DIR}" = "True" ] || [ "${OVERWRITE_OUTPUT_DIR}" = "true" ]; then
    OVERWRITE_FLAG="--overwrite_output_dir"
fi

LOAD_FLAG=""
if [ "$LOAD_MODE" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" = "float16" ]; then
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

    TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-vi${STEP_INTERVAL}-r${RANK}-opt${OPT}-ms${MULTIPLE_SAMPLE}-ns${NUM_SAMPLES}-${RUN_SUFFIX}"
    OUTPUT_DIR="${LLM_DIR}/result_uv/${TRAINER}/${TASK}-${MODEL_NAME}-${TAG}"
    TASK_WANDB_NAME="${TASK}-${MODEL_NAME}-${TAG}"

    echo "Model: ${MODEL}"
    echo "Task: ${TASK}"
    echo "GPU(s): ${GPU}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Trainer: ${TRAINER}"
    echo "Optimizer: ${OPT}"
    echo "Rank: ${RANK}"
    echo "Step interval: ${STEP_INTERVAL}"
    echo "Multiple sample: ${MULTIPLE_SAMPLE}"
    echo "Num samples: ${NUM_SAMPLES}"
    echo "Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"

    WANDB_PROJECT="${WANDB_PROJECT_NAME}" WANDB_NAME="${TASK_WANDB_NAME}" CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
        --model_name "${MODEL}" \
        --task_name "${TASK}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "${TAG}" \
        --train_set_seed "${SEED}" \
        --num_train "${TRAIN}" \
        --num_eval "${EVAL}" \
        --logging_steps 100 \
        --max_steps "${STEPS}" \
        --trainer "${TRAINER}" \
        ${LOAD_FLAG} \
        --zo_optimizer "${OPT}" \
        --multiple_sample "${MULTIPLE_SAMPLE}" \
        --max_time "${MAX_TIME}" \
        --adam_eps 1e-6 \
        --num_samples "${NUM_SAMPLES}" \
        --learning_rate "${LR}" \
        --zo_eps "${EPS}" \
        --per_device_train_batch_size "${BS}" \
        --lr_scheduler_type constant \
        --evaluation_strategy steps \
        --save_strategy steps \
        --save_total_limit 1 \
        --eval_steps "${EVAL_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --step_interval "${STEP_INTERVAL}" \
        --rank_r "${RANK}" \
        --report_to "${REPORT_TO}" \
        ${OVERWRITE_FLAG} \
        ${TASK_ARGS} \
        ${EXTRA_ARGS} \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
