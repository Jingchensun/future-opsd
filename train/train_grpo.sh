#!/bin/bash
# GRPO on the OPSD training data (Future-L1-50K), the outcome-reward counterpart of OPSD (train/train_opsd.sh).
#
#   policy  Qwen3-VL-8B-Instruct, same system prompt / question frame / data as OPSD; text-only reasoning
#   reward  0.9 * correct + 0.1 * format (<reason>...</reason><answer>...</answer>); repetitive output -> correct = -1
#           correct = Qwen3.6-27B (thinking off) judging the <answer> text against the dataset's open-ended answer
#           (RL_v2/future_l1_rl/judge/api_judge.py, yes/no); the judge runs on the same node as a vLLM server
#   GRPO    n=8 rollouts at T=1.0, group-normalised advantage, no KL, PPO clip 0.2/0.28, lr 1e-6 (as OPSD)
#
# GPUs: the judge takes the last JUDGE_NGPU cards (TP), training the rest. Defaults for 8 cards: judge 2, train 6,
# 120 prompts x 8 = 960 rollouts per step, 2 updates of 60 prompts (= 480 rollouts; global_batch_size counts prompts).
#
# Usage: [MODEL=...] [RUN_NAME=...] bash train/train_grpo.sh [extra verl overrides, e.g. trainer.max_steps=2]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
NALL=$(nvidia-smi -L | wc -l); USE_JUDGE=${USE_JUDGE:-1}  # USE_JUDGE=0: rule-based reward (e.g. EgoPlan 4-way MCQ), all cards train
JUDGE_NGPU=$([ $USE_JUDGE = 1 ] && echo ${JUDGE_NGPU:-2} || echo 0); NGPU=$((NALL - JUDGE_NGPU))
TRAIN_GPUS=$(seq -s, 0 $((NGPU - 1))); JUDGE_GPUS=$(seq -s, $NGPU $((NALL - 1)))
JP=${JUDGE_PORT:-18431}
RUN_NAME=${RUN_NAME:-GRPO-judge27b-qwen3vl8b-f50k}
OUT=$CKPT_ROOT/$RUN_NAME; LOG=$OUTPUT_ROOT/grpo; mkdir -p $OUT $LOG
say(){ echo "[grpo $RUN_NAME] $(date -u '+%F %T') $*" | tee -a $LOG/${RUN_NAME}.progress.log; }

# ---- judge server ----
if [ $USE_JUDGE = 1 ]; then
if ! curl -sf localhost:$JP/v1/models >/dev/null; then
  say "starting judge Qwen3.6-27B on GPUs $JUDGE_GPUS (port $JP)"
  CUDA_VISIBLE_DEVICES=$JUDGE_GPUS PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 setsid nohup $ENV_ROOT/judge-vllm/bin/vllm serve \
    $CKPT_ROOT/Qwen3.6-27B --served-model-name Qwen3.6-27B --tensor-parallel-size $JUDGE_NGPU --port $JP \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --default-chat-template-kwargs '{"enable_thinking": false}' \
    > $LOG/${RUN_NAME}.judge_server.log 2>&1 &
  for i in $(seq 1 120); do curl -sf localhost:$JP/v1/models >/dev/null && break; sleep 10; done
fi
curl -sf localhost:$JP/v1/models >/dev/null || { say "judge server failed (see ${RUN_NAME}.judge_server.log)"; exit 1; }
say "judge up"
fi

# TRAIN_ENV / RL_DIR: Qwen3.5 runs use envs/future-l1-q35 (vLLM 0.30, transformers 5) and the RL_v2_q35 port (E25)
ENV=${TRAIN_ENV:-$ENV_ROOT/future-l1-rl-vllm011}
export PATH=$ENV/bin:$PATH PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export FUTURE_L1_ROOT=$ROOT/train/sft FUTURE_L1_CODE_ROOT=$ROOT/train/sft
export HF_HOME=$HF_HOME_EVAL HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DASHBOARD=1 RAY_DASHBOARD_ENABLED=0 USE_RAY_LOCAL=1 RAY_ADDRESS=local
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-64} RAY_NUM_GPUS=$NGPU RAY_SPILL_DIR=${RAY_SPILL_DIR:-${TMPDIR:-/tmp}/ray_spill} RAY_TMPDIR=${RAY_TMPDIR:-${TMPDIR:-/tmp}/ray_tmp}
export FUTURE_L1_RL_PATCH=0 OPSD_MODE=0 OPSD_TEACHER=${OPSD_TEACHER:-0}
# E20a HINT_RESAMPLE=1 (ray_trainer._hint_resample): all-wrong groups are re-sampled from the future view, so the dataset
# also builds that view (OPSD_TEACHER=1, the OPSD teacher prompt with the true future frames); no teacher scoring
[ "${HINT_RESAMPLE:-0}" = 1 ] && export OPSD_TEACHER=1 OPSD_MAX_FUTURE=${OPSD_MAX_FUTURE:-3} OPSD_TEACHER_MAX_PROMPT=${OPSD_TEACHER_MAX_PROMPT:-8192}
export RL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)"  # same prompt as OPSD and the evaluation
export USE_LLM_JUDGE=1 LLM_JUDGE_ONLY=1 JUDGE_API_NAME=Qwen3.6-27B JUDGE_API_URL=http://127.0.0.1:$JP/v1 JUDGE_API_KEY=EMPTY
REWARD_FN=${REWARD_FN:-./examples/reward_function/cot_thinking_reward_function.py:compute_score}
REWARD_KW=(worker.reward.reward_function_kwargs.length_penalty_weight=0.0)
[ $USE_JUDGE = 1 ] || { export USE_LLM_JUDGE=0; REWARD_KW=(); }
export API_JUDGE_WORKERS=${API_JUDGE_WORKERS:-128} COT_FORMAT_WEIGHT=0.1 COT_LENGTH_PENALTY_WEIGHT=0
export WANDB_MODE=offline WANDB_PROJECT=Future-L1-GRPO WANDB_DIR=$OUT/wandb; mkdir -p $WANDB_DIR

say "training on GPUs $TRAIN_GPUS"
# EGOPLAN=1: train on EgoPlan-IT (data_tools/make_egoplan_train.py: progress video + observation image + 4-way question)
# with the SEED-Bench-R1 evaluation's visual settings (lmms-eval qwen3_vl_vllm: 32 uniformly sampled video frames,
# every frame / image at >= 256*28*28 pixels); the rollout engine then also takes one video per prompt
if [ "${EGOPLAN:-0}" = 1 ]; then
  DATA_ARGS=(data.train_files=$DATA_ROOT/egoplan/train/egoplan_train.jsonl data.max_prompt_length=6144
    data.video_fps=${EGO_VIDEO_FPS:-30.0} data.max_video_frames=32
    data.min_pixels=200704 data.max_pixels=1605632 data.image_min_pixels=200704 data.image_max_pixels=1605632
    data.video_min_pixels=200704 data.video_max_pixels=1605632)
  export ROLLOUT_LIMIT_VIDEOS=1
else
  DATA_ARGS=(data.train_files=$DATA_ROOT/future_l1/train/Future-L1-50K.opsd.json data.max_prompt_length=2048
    data.image_max_pixels=1048576 data.max_pixels=1048576)
fi
cd $ROOT/${RL_DIR:-train/rl}
python -m verl.trainer.main \
  config=examples/config_future_l1.yaml \
  "${DATA_ARGS[@]}" data.max_response_length=1024 \
  data.rollout_batch_size=${ROLLOUT_BATCH:-120} data.mini_rollout_batch_size=${ROLLOUT_BATCH:-120} \
  algorithm.adv_estimator=grpo algorithm.online_filtering=false algorithm.answer_tag_filtering=false \
  algorithm.disable_kl=true algorithm.use_kl_loss=false algorithm.kl_coef=0.0 \
  algorithm.enable_decoupled_hybrid_ppo=false \
  worker.actor.model.model_path=${MODEL:-$CKPT_ROOT/Qwen3-VL-8B-Instruct} \
  worker.actor.global_batch_size=${GLOBAL_BATCH:-60} worker.actor.optim.lr=${LR:-1.0e-6} \
  worker.rollout.n=8 worker.rollout.temperature=1.0 worker.rollout.top_p=1.0 \
  worker.rollout.sampling_strategy=default worker.rollout.limit_images=4 \
  worker.reward.reward_function=$REWARD_FN "${REWARD_KW[@]}" \
  worker.reward.skip_special_tokens=true \
  trainer.n_gpus_per_node=$NGPU trainer.experiment_name=$RUN_NAME trainer.project_name=Future-L1-GRPO \
  trainer.total_epochs=1 trainer.max_steps=${MAX_STEPS:-100} trainer.save_freq=${SAVE_FREQ:-50} trainer.save_limit=3 \
  trainer.save_checkpoint_path=$OUT/checkpoints trainer.save_samples=true trainer.samples_save_dir=$OUT/rollout_samples \
  trainer.samples_save_interval=10 \
  "$@" 2>&1 | tee -a $LOG/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log
rc=${PIPESTATUS[0]}
[ "${KEEP_JUDGE:-0}" = 1 ] || [ $USE_JUDGE = 0 ] || pkill -f "vllm serve $CKPT_ROOT/Qwen3.6-27B" 2>/dev/null
say "training exit=$rc"
exit $rc
