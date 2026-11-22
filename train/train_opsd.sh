#!/bin/bash
# On-policy self-distillation (OPSD) with privileged future frames, as a replacement for Future-L1's SFT.
#
#   student  pi(. | x)      x = question frame(s) + question (what the model sees at test time)
#   teacher  pi_0(. | x, z) the FROZEN initial model (reference worker). OPSD_TEACHER_WEIGHTS=ema: EMA of the student
#                           (decay OPSD_EMA_DECAY, default 0.98 per step); =current: the live policy (collapses);
#                           z = the sample's future frames (reasoning_image, <=3)
#   rollout: the student samples its own reasoning + answer; the teacher scores those tokens;
#            per-token advantage = log pi_teacher - log pi_student (sampled estimator of -reverse KL),
#            optimised with the PPO clipped objective (verl/trainer/ray_trainer.py, OPSD_MODE=1).
#
# Data: Future-L1-50K (the same data as the Future-L1 SFT). Start: Qwen3-VL-8B-Instruct. Text-only reasoning.
# Control: OPSD_WRONG_FUTURE=1 gives the teacher the future frames of a different video.
# Privileged context: OPSD_PRIVILEGED=frames (default) | answer (the dataset answer text) | text (the reference
#   reasoning's description of frame_2 onwards) | frames+answer. answer/text give the teacher no future frames.
#
# Usage: [MODEL=<base model dir, default Qwen3-VL-8B-Instruct>] [OPSD_WRONG_FUTURE=1] [RUN_NAME=...] bash train/train_opsd.sh [extra verl overrides]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
REWARD_FN=./examples/reward_function/opsd_logging_reward.py:compute_score; REWARD_KW=()
if [ "${OPSD_OUTCOME_LAMBDA:-0}" != 0 ] && [ "${OPSD_OUTCOME_REWARD:-judge}" = egoplan ]; then
  # E19: + lambda * the GRPO advantage of the EgoPlan rule reward (as E18b: 0.9 * option letter correct + 0.1 * format),
  # no judge server, all cards train
  REWARD_FN=./examples/reward_function/egoplan_mcq_reward.py:compute_score
elif [ "${OPSD_OUTCOME_LAMBDA:-0}" != 0 ]; then
  # E17: OPSD + lambda * GRPO advantage of the judge reward (as train/train_grpo.sh: 0.9 * correct + 0.1 * format,
  # correct = Qwen3.6-27B judging the <answer> against the dataset answer). The judge takes the last JUDGE_NGPU cards.
  NALL=$NGPU; JUDGE_NGPU=${JUDGE_NGPU:-2}; NGPU=$((NALL - JUDGE_NGPU)); JP=${JUDGE_PORT:-18431}
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1)))
  JLOG=$OUTPUT_ROOT/opsd/${RUN_NAME:-opsd}.judge_server.log
  if ! curl -sf localhost:$JP/v1/models >/dev/null; then
    CUDA_VISIBLE_DEVICES=$(seq -s, $NGPU $((NALL - 1))) PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 setsid nohup $ENV_ROOT/judge-vllm/bin/vllm serve \
      $CKPT_ROOT/Qwen3.6-27B --served-model-name Qwen3.6-27B --tensor-parallel-size $JUDGE_NGPU --port $JP \
      --max-model-len 8192 --gpu-memory-utilization 0.85 --default-chat-template-kwargs '{"enable_thinking": false}' \
      > $JLOG 2>&1 &
    for i in $(seq 1 120); do curl -sf localhost:$JP/v1/models >/dev/null && break; sleep 10; done
  fi
  curl -sf localhost:$JP/v1/models >/dev/null || { echo "judge server failed (see $JLOG)"; exit 1; }
  export USE_LLM_JUDGE=1 LLM_JUDGE_ONLY=1 JUDGE_API_NAME=Qwen3.6-27B JUDGE_API_URL=http://127.0.0.1:$JP/v1 JUDGE_API_KEY=EMPTY
  export API_JUDGE_WORKERS=${API_JUDGE_WORKERS:-128} COT_FORMAT_WEIGHT=0.1 COT_LENGTH_PENALTY_WEIGHT=0
  REWARD_FN=./examples/reward_function/cot_thinking_reward_function.py:compute_score
  REWARD_KW=(worker.reward.reward_function_kwargs.length_penalty_weight=0.0)
fi
# TRAIN_ENV / RL_DIR: Qwen3.5 runs use envs/future-l1-q35 (vLLM 0.30, transformers 5) and the RL_v2_q35 port (E25)
ENV=${TRAIN_ENV:-$ENV_ROOT/future-l1-rl-vllm011}
export PATH=$ENV/bin:$PATH PYTHONNOUSERSITE=1
unset PYTHONPATH
export FUTURE_L1_ROOT=$ROOT/train/sft FUTURE_L1_CODE_ROOT=$ROOT/train/sft
export HF_HOME=$HF_HOME_EVAL HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DASHBOARD=1 RAY_DASHBOARD_ENABLED=0 USE_RAY_LOCAL=1 RAY_ADDRESS=local
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-64} RAY_NUM_GPUS=$NGPU RAY_SPILL_DIR=${RAY_SPILL_DIR:-${TMPDIR:-/tmp}/ray_spill} RAY_TMPDIR=${RAY_TMPDIR:-${TMPDIR:-/tmp}/ray_tmp}

export FUTURE_L1_RL_PATCH=0            # plain Qwen3-VL: no latent tokens, stock vLLM rollout
export OPSD_MODE=1 OPSD_TEACHER=1 OPSD_MAX_FUTURE=${OPSD_MAX_FUTURE:-3} OPSD_TEACHER_MAX_PROMPT=${OPSD_TEACHER_MAX_PROMPT:-$([ "${EGOPLAN:-0}" = 1 ] && echo 8192 || echo 6144)} OPSD_ADV_CLIP=${OPSD_ADV_CLIP:-5.0}
export OPSD_WRONG_FUTURE=${OPSD_WRONG_FUTURE:-0}
export OPSD_PRIVILEGED=${OPSD_PRIVILEGED:-frames}
case $OPSD_PRIVILEGED in answer|text) export OPSD_MAX_FUTURE=0;; esac  # no future frames to decode
# teacher = frozen initial model + future frames (reference worker); advantages centred by their batch mean
export OPSD_TEACHER_WEIGHTS=${OPSD_TEACHER_WEIGHTS:-ref} OPSD_ADV_BASELINE=${OPSD_ADV_BASELINE:-batch_mean}
export RL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)"  # the same prompt is used at evaluation

TAG=$([ "$OPSD_WRONG_FUTURE" = 1 ] && echo wrongfuture || echo future)
RUN_NAME=${RUN_NAME:-OPSD-${TAG}-qwen3vl8b-f50k}
OUT=$CKPT_ROOT/$RUN_NAME; LOG=$OUTPUT_ROOT/opsd; mkdir -p $OUT $LOG
export WANDB_MODE=offline WANDB_PROJECT=Future-L1-OPSD WANDB_DIR=$OUT/wandb; mkdir -p $WANDB_DIR

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
  data.rollout_batch_size=${ROLLOUT_BATCH:-256} data.mini_rollout_batch_size=${ROLLOUT_BATCH:-256} \
  algorithm.online_filtering=false algorithm.answer_tag_filtering=false \
  algorithm.disable_kl=false algorithm.use_kl_loss=true algorithm.kl_coef=0.0 worker.ref.fsdp.enable_cpu_offload=false \
  algorithm.enable_decoupled_hybrid_ppo=false \
  worker.actor.model.model_path=${MODEL:-$CKPT_ROOT/Qwen3-VL-8B-Instruct} \
  worker.actor.global_batch_size=${ACTOR_GBS:-128} worker.actor.optim.lr=${LR:-1.0e-6} \
  worker.rollout.n=${ROLLOUT_N:-1} worker.rollout.temperature=1.0 worker.rollout.top_p=1.0 \
  worker.rollout.sampling_strategy=default worker.rollout.limit_images=4 \
  worker.reward.reward_function=$REWARD_FN "${REWARD_KW[@]}" \
  worker.reward.skip_special_tokens=true \
  trainer.n_gpus_per_node=$NGPU trainer.experiment_name=$RUN_NAME trainer.project_name=Future-L1-OPSD \
  trainer.total_epochs=1 trainer.max_steps=${MAX_STEPS:-195} trainer.save_freq=${SAVE_FREQ:-65} trainer.save_limit=3 \
  trainer.save_checkpoint_path=$OUT/checkpoints trainer.save_samples=true trainer.samples_save_dir=$OUT/rollout_samples \
  trainer.samples_save_interval=25 \
  "$@" 2>&1 | tee -a $LOG/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log
rc=${PIPESTATUS[0]}
[ "${OPSD_OUTCOME_LAMBDA:-0}" != 0 ] && [ "${OPSD_OUTCOME_REWARD:-judge}" != egoplan ] && [ "${KEEP_JUDGE:-0}" != 1 ] && pkill -f "vllm serve $CKPT_ROOT/Qwen3.6-27B" 2>/dev/null
exit $rc
