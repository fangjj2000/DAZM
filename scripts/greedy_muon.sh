#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-facebook/opt-13b}"  # Qwen/Qwen2.5-32B  facebook/opt-13b  Qwen/Qwen2.5-7B  meta-llama/Meta-Llama-3-8B
DEFAULT_EARLY_STOPPING_PATIENCE=0
DEFAULT_EARLY_STOPPING_MIN_STEPS=0
if [ "${MODEL}" = "Qwen/Qwen2.5-7B" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
    DEFAULT_EARLY_STOPPING_PATIENCE=3
    DEFAULT_EARLY_STOPPING_MIN_STEPS=2000
fi
MODE="${MODE:-ft}"
GPU="${GPU:-1}"
BS="${BS:-16}"
EPS="${EPS:-1e-3}"
LR="${LR:-1e-2}"
STEPS="${STEPS:-6000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-steps}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
TRAIN="${TRAIN:-1000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LOAD_MODE="${LOAD_MODE:-float16}"
MAX_TIME="${MAX_TIME:-0}"
SEED="${SEED:-0}"
REPORT_TO="${REPORT_TO:-none}"
SAVE_ON_INTERRUPT="${SAVE_ON_INTERRUPT:-True}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-gemma}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-True}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"
PERTURBATION_MODE="${PERTURBATION_MODE:-one_side}"

RANK="${RANK:-256}"
ENERGY_THRESHOLD="${ENERGY_THRESHOLD:-0.9}"
P="${P:-256}"
STEP_INTERVAL="${STEP_INTERVAL:-100}"
SUBSPACE_UPDATE_MODE="${SUBSPACE_UPDATE_MODE:-gradient}"
CAPTURE_DISPLACEMENT="${CAPTURE_DISPLACEMENT:-False}"
CAPTURE_LAYERS="${CAPTURE_LAYERS:-}"
CAPTURE_REFRESH_STRIDE="${CAPTURE_REFRESH_STRIDE:-5}"
CAPTURE_OUTPUT_FILE="${CAPTURE_OUTPUT_FILE:-}"
OPT="${OPT:-muon}"
MULTIPLE_SAMPLE="${MULTIPLE_SAMPLE:-True}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MOMENTUM="${MOMENTUM:-False}"
MOMENTUM_BETA="${BETA:-0.9}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-${DEFAULT_EARLY_STOPPING_PATIENCE}}"
EARLY_STOPPING_MIN_STEPS="${EARLY_STOPPING_MIN_STEPS:-${DEFAULT_EARLY_STOPPING_MIN_STEPS}}"
EARLY_STOPPING_METRIC_DROP="${EARLY_STOPPING_METRIC_DROP:-0.02}"
EARLY_STOPPING_LOSS_RATIO="${EARLY_STOPPING_LOSS_RATIO:-1.15}"

# TASK=SST2 | TASKS="SST2 RTE Copa"  # SST2 RTE CB BoolQ WSC WIC MultiRC Copa ReCoRD SQuAD DROP
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="greedyzomuon"
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

SAVE_ON_INTERRUPT_FLAG=""
if [ "${SAVE_ON_INTERRUPT}" = "True" ] || [ "${SAVE_ON_INTERRUPT}" = "true" ]; then
    SAVE_ON_INTERRUPT_FLAG="--save_on_interrupt"
fi

RESULT_FILE_FLAG=()
if [ -n "${RESULT_FILE:-}" ]; then
    RESULT_FILE_FLAG=(--result_file "${RESULT_FILE}")
fi

CAPTURE_FLAG=()
if [ "${CAPTURE_DISPLACEMENT}" = "True" ] || [ "${CAPTURE_DISPLACEMENT}" = "true" ]; then
    CAPTURE_FLAG+=(--capture_displacement True)
    CAPTURE_FLAG+=(--capture_refresh_stride "${CAPTURE_REFRESH_STRIDE}")
    [ -n "${CAPTURE_LAYERS}" ] && CAPTURE_FLAG+=(--capture_layers "${CAPTURE_LAYERS}")
    [ -n "${CAPTURE_OUTPUT_FILE}" ] && CAPTURE_FLAG+=(--capture_output_file "${CAPTURE_OUTPUT_FILE}")
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
            # CB has only about 150 training examples; DEV=500 would leave no
            # samples for the training dataloader. Keep a disjoint 100-example dev split.
            DEV=100
            ;;
        Copa)
            # Copa has only about 300 training examples; DEV=500 is infeasible.
            DEV=100
            TASK_ARGS="--train_as_classification False"
            ;;
        ReCoRD|DROP|SQuAD)
            TASK_ARGS="--train_as_classification False"
            ;;
    esac

    TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-${PERTURBATION_MODE}-vi${STEP_INTERVAL}-r${RANK}-energy${ENERGY_THRESHOLD}-subspace${SUBSPACE_UPDATE_MODE}-opt${OPT}-ms${MULTIPLE_SAMPLE}-ns${NUM_SAMPLES}-mom${MOMENTUM}-b${MOMENTUM_BETA}-es${EARLY_STOPPING_PATIENCE}-${RUN_SUFFIX}"
    OUTPUT_DIR="${LLM_DIR}/result_uv/${TRAINER}/${TASK}-${MODEL_NAME}-${TAG}"
    TASK_WANDB_NAME="${TASK}-${MODEL_NAME}-${TAG}"

    echo "Model: ${MODEL}"
    echo "Task: ${TASK}"
    echo "GPU(s): ${GPU}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Trainer: ${TRAINER}"
    echo "Optimizer: ${OPT}"
    echo "Train/Dev/Eval: ${TRAIN}/${DEV}/${EVAL}"
    echo "Logging steps: ${LOGGING_STEPS}"
    echo "Evaluation/save strategy: ${EVALUATION_STRATEGY}/${SAVE_STRATEGY}"
    echo "Eval steps: ${EVAL_STEPS}"
    echo "Checkpoint policy: keep checkpoint-best; also keep last if last != best"
    echo "Rank: ${RANK}"
    echo "Step interval: ${STEP_INTERVAL}"
    echo "Subspace update mode: ${SUBSPACE_UPDATE_MODE}"
    echo "Multiple sample: ${MULTIPLE_SAMPLE}"
    echo "Num samples: ${NUM_SAMPLES}"
    echo "Momentum/Beta: ${MOMENTUM}/${MOMENTUM_BETA}"
    echo "Early stopping: patience=${EARLY_STOPPING_PATIENCE}, min_steps=${EARLY_STOPPING_MIN_STEPS}, metric_drop=${EARLY_STOPPING_METRIC_DROP}, loss_ratio=${EARLY_STOPPING_LOSS_RATIO}"
    echo "Perturbation mode: ${PERTURBATION_MODE}"
    echo "Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"

    WANDB_PROJECT="${WANDB_PROJECT_NAME}" WANDB_NAME="${TASK_WANDB_NAME}" CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
        --model_name "${MODEL}" \
        --task_name "${TASK}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "${TAG}" \
        --train_set_seed "${SEED}" \
        --num_train "${TRAIN}" \
        --num_dev "${DEV}" \
        --num_eval "${EVAL}" \
        --logging_steps "${LOGGING_STEPS}" \
        --max_steps "${STEPS}" \
        --trainer "${TRAINER}" \
        ${LOAD_FLAG} \
        --zo_optimizer "${OPT}" \
        --multiple_sample "${MULTIPLE_SAMPLE}" \
        --momentum "${MOMENTUM}" \
        --beta "${MOMENTUM_BETA}" \
        --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
        --early_stopping_min_steps "${EARLY_STOPPING_MIN_STEPS}" \
        --early_stopping_metric_drop "${EARLY_STOPPING_METRIC_DROP}" \
        --early_stopping_loss_ratio "${EARLY_STOPPING_LOSS_RATIO}" \
        --max_time "${MAX_TIME}" \
        --adam_eps 1e-6 \
        --num_samples "${NUM_SAMPLES}" \
        --learning_rate "${LR}" \
        --zo_eps "${EPS}" \
        --per_device_train_batch_size "${BS}" \
        --lr_scheduler_type constant \
        --evaluation_strategy "${EVALUATION_STRATEGY}" \
        --save_strategy "${SAVE_STRATEGY}" \
        --metric_for_best_model accuracy \
        --greater_is_better True \
        --eval_steps "${EVAL_STEPS}" \
        --step_interval "${STEP_INTERVAL}" \
        --subspace_update_mode "${SUBSPACE_UPDATE_MODE}" \
        --rank_r "${RANK}" \
        --p "${P}" \
        --energy_threshold "${ENERGY_THRESHOLD}" \
        --zo_perturbation_mode "${PERTURBATION_MODE}" \
        --report_to "${REPORT_TO}" \
        ${SAVE_ON_INTERRUPT_FLAG} \
        ${OVERWRITE_FLAG} \
        ${TASK_ARGS} \
        ${EXTRA_ARGS} \
        "${RESULT_FILE_FLAG[@]}" \
        "${CAPTURE_FLAG[@]}" \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
