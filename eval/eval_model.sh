#!/bin/bash
# Evaluate one model on FutureBench + TwiFF-Bench (2 x B200 or more).
#
#   bash eval/eval_model.sh <TAG> <MODEL_PATH> <KIND> [STAGES]
#     KIND   base      -> lmms-eval `qwen3_vl` wrapper (plain Qwen3-VL, zero-shot)
#            opsd      -> `qwen3_vl` with the OPSD system prompt (config/opsd_system_prompt.txt)
#            future_l1 -> lmms-eval `future_l1` wrapper (latent-aware generation)
#     STAGES comma list, default "fb,twiff,judge"
#
# Outputs: output/<TAG>/{futurebench*,twiffbench,logs}/ and output/<TAG>/summary.json
#
# TwiFF-Bench is generated with TWIFF_SKIP_JUDGE=1, then scored by a local Qwen3.6-27B
# (vLLM, TP=2, thinking disabled) through lmms-eval/tools/twiffbench/post_score.py.
set -uo pipefail
# LAUNCH_TIMEOUT: kill a model launch after this many seconds (default 90 min). lmms-eval's NCCL timeout is
# 60000 s, so a rank stuck in the final gather otherwise hangs the job for hours with nvidia-smi showing 100%
# (spinning NCCL kernels); the timeout turns such a hang into a retry.
TAG=${1:?tag}; MODEL=${2:?model path}; KIND=${3:?base|future_l1}; STAGES=${4:-fb,twiff,judge}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
OUT=$OUTPUT_ROOT/$TAG; LOG=$OUT/logs; mkdir -p $LOG
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PORT=${PORT:-$((12000 + RANDOM % 1000))}

# BACKEND=vllm (KIND base|opsd only): lmms-eval `qwen3_vl_vllm`, one vLLM engine per GPU, same prompts/preprocessing
BACKEND=${BACKEND:-hf}
# VLLM_ENV: vLLM env for the model (default future-l1-rl; future-l1-q35 = vLLM 0.30 / transformers 5 for Qwen3.5);
# VLLM_MA_EXTRA: appended to the vLLM model_args (Qwen3.5: ",enable_thinking=False" keeps the reason-then-answer protocol)
if [ "$BACKEND" = vllm ]; then export PATH=$ENV_ROOT/${VLLM_ENV:-future-l1-rl}/bin:$PATH PYTHONNOUSERSITE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
else export PATH=$ENV_ROOT/future-l1-sft/bin:$PATH PYTHONNOUSERSITE=1; fi
unset PYTHONPATH http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export FUTURE_L1_ROOT=$ROOT/train/sft FUTURE_L1_CODE_ROOT=$ROOT/train/sft VIDEO_L1_ROOT=$ROOT/train/sft
export HF_HOME=$HF_HOME_EVAL HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENCV_FFMPEG_LOGLEVEL=-8 AV_LOG_LEVEL=quiet
say(){ echo "[eval $TAG] $(date -u '+%F %T') $*" | tee -a $LOG/progress.log; }
has(){ [[ ",$STAGES," == *",$1,"* ]]; }
source $ROOT/utils/gpu_clean.sh

if [ "$KIND" = base ]; then
  M=qwen3_vl; MA="pretrained=$MODEL,attn_implementation=flash_attention_2"
  FB_TASKS=(futurebench_thinking futurebench); TW_TASK=twiffbench; STRIP=--no-strip-latent
elif [ "$KIND" = opsd ]; then   # plain Qwen3-VL trained by OPSD: its training system prompt, <reason>/<answer> output
  M=qwen3_vl; MA="pretrained=$MODEL,attn_implementation=flash_attention_2"
  export QWEN3VL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)${OPSD_SYSTEM_SUFFIX:-}"
  FB_TASKS=(futurebench_future_l1); TW_TASK=twiffbench_future_l1; STRIP=--strip-latent
elif [ "$KIND" = cvae ]; then   # E14 CVAE (cvae/export.py dir): prior latent slots + OPSD prompt, HF only
  M=qwen3_vl_cvae; MA="pretrained=$MODEL,attn_implementation=flash_attention_2,latent=${CVAE_LATENT:-mean}"
  export QWEN3VL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)${OPSD_SYSTEM_SUFFIX:-}"
  FB_TASKS=(futurebench_future_l1); TW_TASK=twiffbench_future_l1; STRIP=--strip-latent
else
  M=future_l1; MA="pretrained=$MODEL,code_root=$ROOT,attn_implementation=flash_attention_2"
  FB_TASKS=(futurebench_future_l1); TW_TASK=twiffbench_future_l1; STRIP=--strip-latent
fi
if [ "$BACKEND" = vllm ]; then
  case $KIND in future_l1|cvae) echo "BACKEND=vllm does not support KIND=$KIND"; exit 1;; esac
  M=qwen3_vl_vllm; MA="pretrained=$MODEL${VLLM_MA_EXTRA:-}"; BATCH_FB=${VLLM_BATCH:-256}; BATCH_TW=${VLLM_BATCH:-256}
fi

run_lmms(){  # $1=task $2=out subdir $3=max_num_frames $4=max_new_tokens $5=per-GPU batch size
  if ls $OUT/$2/*/*_results.json >/dev/null 2>&1 && ls $OUT/$2/*/*samples_$1*.jsonl >/dev/null 2>&1; then say "lmms-eval $1 already done, skip"; return 0; fi
  gpu_ready; say "lmms-eval $1 (frames=$3, max_new_tokens=$4) ..."
  ( cd $ROOT/eval/lmms-eval && timeout -k 120 ${LAUNCH_TIMEOUT:-5400} accelerate launch --num_processes=$NGPU --main_process_port=$PORT -m lmms_eval \
      --model $M --model_args "$MA,max_num_frames=$3" --tasks $1 --batch_size $5 \
      --gen_kwargs max_new_tokens=$4,temperature=0,top_p=1,do_sample=false \
      --log_samples --log_samples_suffix $TAG --output_path $OUT/$2 ${LIMIT:+--limit $LIMIT} ) > $LOG/$2.log 2>&1
  local rc=$?
  # lmms-eval can swallow task errors and still exit 0, so check that a results file was written
  if ls $OUT/$2/*/*_results.json >/dev/null 2>&1; then say "lmms-eval $1 OK"; else say "lmms-eval $1 FAILED (exit=$rc, no results.json; see logs/$2.log)"; rc=1; fi
  return $rc
}

[ -n "${FB_TASKS_OVERRIDE:-}" ] && FB_TASKS=($FB_TASKS_OVERRIDE)   # e.g. FB_TASKS_OVERRIDE=futurebench_shuf
if has fb; then
  for t in "${FB_TASKS[@]}"; do run_lmms $t $t 32 2048 ${BATCH_FB:-16}; done
fi
if has twiff; then
  TWIFF_SKIP_JUDGE=1 run_lmms $TW_TASK twiffbench 8 4096 ${BATCH_TW:-32}
fi

if has judge; then
  SAMPLES=$(ls -t $OUT/twiffbench/*/*samples_${TW_TASK}*.jsonl 2>/dev/null | grep -v scored | head -1)
  if [ -z "$SAMPLES" ]; then say "judge: no twiffbench samples found"
  elif ls ${SAMPLES%.jsonl}.scored*.summary.json >/dev/null 2>&1; then say "judge: already scored, skip"; else
    gpu_ready; say "judge: starting Qwen3.6-27B vLLM server (TP=$NGPU)"
    JP=$((PORT + 1000))
    PYTHONNOUSERSITE=1 $ENV_ROOT/judge-vllm/bin/vllm serve $CKPT_ROOT/Qwen3.6-27B --served-model-name Qwen3.6-27B \
      --tensor-parallel-size $NGPU --port $JP --max-model-len 32768 --gpu-memory-utilization 0.85 \
      --default-chat-template-kwargs '{"enable_thinking": false}' > $LOG/judge_server.log 2>&1 &
    JPID=$!
    for i in $(seq 1 120); do curl -sf localhost:$JP/v1/models >/dev/null && break; sleep 10; done
    if curl -sf localhost:$JP/v1/models >/dev/null; then
      say "judge: server up, scoring $SAMPLES"
      ( cd $ROOT/eval/lmms-eval && OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://127.0.0.1:$JP/v1 LOCAL_LLM=Qwen3.6-27B \
          python tools/twiffbench/post_score.py --input "$SAMPLES" --concurrency 64 $STRIP --parquet $DATA_ROOT/future_l1/test/test-0000{0,1,2,3}-of-00004.parquet ) > $LOG/judge_score.log 2>&1
      say "judge: exit=$?"
    else say "judge: server failed to start (see logs/judge_server.log)"; fi
    kill $JPID 2>/dev/null; sleep 5; pkill -9 -f "vllm serve $CKPT_ROOT/Qwen3.6-27B" 2>/dev/null; gpu_ready
  fi
fi

python $ROOT/eval/summarize.py $OUT | tee -a $LOG/progress.log
say "ALL DONE"
