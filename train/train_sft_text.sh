#!/bin/bash
# Plain-text SFT on Future-L1-50K: the SFT arm of the SFT / OPSD / GRPO comparison.
# Hyper-parameters of the released Future-L1 SFT recipe (lr 1e-5 cosine, warmup 0.1, 1 epoch, global batch 128,
# frozen ViT + merger), but every sample uses the text-only CoT format (twiff_cot_ratio=1.0): the future frames are
# dropped and the target is "<reason>{dataset reasoning text}</reason>\n<answer>{answer}</answer>", with the OPSD
# system prompt -- the same input/output format as OPSD, GRPO and the evaluation. No latent tokens, latent_lambda 0.
#
# Usage: bash train/train_sft_text.sh [extra HF args]    (output: ckpt/$RUN_NAME, log: output/sft/)
#   smoke: RUN_NAME=_smoke_sft_text bash train/train_sft_text.sh --max_steps 2 --save_strategy no
#   Qwen3.5-9B (E25): SFT_ENV=future-l1-q35 SFT_MODEL=$CKPT_ROOT/Qwen3.5-9B DISABLE_FA2=True WARMUP_ARGS="--warmup_steps 0.1"
#   (transformers 5 dropped --warmup_ratio; a warmup_steps value < 1 is a ratio. That env has no flash-attn -> sdpa)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
export PATH=$ENV_ROOT/${SFT_ENV:-future-l1-sft}/bin:$PATH PYTHONNOUSERSITE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
SFT=$ROOT/train/sft   # the SFT code (python package `src`)
export PYTHONPATH=$SFT FUTURE_L1_ROOT=$SFT VIDEO_L1_ROOT=$SFT FUTURE_L1_CODE_ROOT=$SFT
export OPENCV_FFMPEG_LOGLEVEL=-8 AV_LOG_LEVEL=quiet TOKENIZERS_PARALLELISM=false
export HF_HOME=$HF_HOME_EVAL HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_PROJECT=Future-L1

NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
GLOBAL_BATCH_SIZE=128; BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-16}   # 8 GPU x 16 = 128 -> no grad accumulation
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NGPU)))
MODEL_NAME=${SFT_MODEL:-$CKPT_ROOT/Qwen3-VL-8B-Instruct-futurel1}         # base weights + repo chat_template.json
DATA_PATH=$DATA_ROOT/future_l1/train/Future-L1-50K.videos.json
LATENT_LAMBDA=0.0; MAX_LATENT_TOKEN=2
RUN_NAME=${RUN_NAME:-SFT-text-qwen3vl8b-f50k}
OUTPUT_DIR=$CKPT_ROOT/$RUN_NAME; LOG_DIR=$OUTPUT_ROOT/sft
export WANDB_DIR=$OUTPUT_DIR
mkdir -p $OUTPUT_DIR $LOG_DIR
cd $SFT

torchrun --nproc_per_node $NGPU --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port ${MASTER_PORT:-29502} \
    src/train/train.py \
    --run_name "$RUN_NAME" \
    --deepspeed $ROOT/config/deepspeed/zero2.json \
    --latent_loss mse \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate 1e-5 \
    --latent_lambda $LATENT_LAMBDA \
    --max_latent_token $MAX_LATENT_TOKEN \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 ${DISABLE_FA2:-False} \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((2 * 32 * 32)) \
    --image_max_pixels $((16384 * 32 * 32)) \
    --video_max_pixels $((128 * 32 * 32)) \
    --nframes 16 \
    --weight_decay 0.1 \
    ${WARMUP_ARGS:---warmup_ratio 0.1} \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --tf32 False \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --dataloader_num_workers ${NUM_WORKERS:-3} \
    --random_seed 42 \
    --use_twiff_dataset True \
    --twiff_cot_ratio 1.0 \
    --system_message_mode custom \
    --custom_system_message_path $ROOT/config/opsd_system_prompt.txt \
    --report_to wandb "$@" 2>&1 | tee -a $LOG_DIR/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log
