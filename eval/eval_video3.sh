#!/bin/bash
# SEED-Bench-R1 / TempCompass / TemporalBench for one model, with the task configs and data of
# our earlier evaluation code (same subtasks, same scoring) so numbers are comparable to that project.
#
#   bash eval/eval_video3.sh <TAG> <MODEL_PATH> <KIND>
#     KIND base      -> `qwen3_vl`, direct-answer prompts and each task's own max_new_tokens (as our earlier evaluation code)
#          future_l1 -> `future_l1`; the model reasons (<reason>/latent) before <answer>, so max_new_tokens=2048 and
#                       extract_answer=True hands only the <answer>...</answer> content to the task's parser
#
# Output: output/<TAG>/{seedbench_r1/l1..l3, tempcompass/<sub>, temporalbench/<sub>}/{results.json,samples.jsonl}
# Scores: python eval/score_video3.py output/<TAG>
set -uo pipefail
# LAUNCH_TIMEOUT: kill a model launch after this many seconds (default 90 min). lmms-eval's NCCL timeout is
# 60000 s, so a rank stuck in the final gather otherwise hangs the job for hours with nvidia-smi showing 100%
# (spinning NCCL kernels); the timeout turns such a hang into a retry.
TAG=${1:?tag}; MODEL=${2:?model path}; KIND=${3:?base|future_l1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
OUT=$OUTPUT_ROOT/$TAG; LOG=$OUT/logs; mkdir -p $LOG
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PORT=${PORT:-$((13000 + RANDOM % 1000))}
BATCH=${BATCH:-16}

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
say(){ echo "[video3 $TAG] $(date -u '+%F %T') $*" | tee -a $LOG/progress.log; }
source $ROOT/utils/gpu_clean.sh

if [ "$KIND" = base ]; then
  M=qwen3_vl; MA="pretrained=$MODEL,attn_implementation=flash_attention_2,max_num_frames=32"; GEN=()
elif [ "$KIND" = opsd ]; then
  M=qwen3_vl; MA="pretrained=$MODEL,attn_implementation=flash_attention_2,max_num_frames=32,extract_answer=True"
  [ -n "${REASONING_BUDGET:-}" ] && MA="$MA,reasoning_budget=$REASONING_BUDGET"   # HF budget forcing (qwen3_vl BudgetForcing)
  export QWEN3VL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)${OPSD_SYSTEM_SUFFIX:-}"
  GEN=(--gen_kwargs max_new_tokens=${MAX_NEW_TOKENS:-1024},temperature=0,top_p=1,do_sample=false)  # was 2048 before 2026-09-27
elif [ "$KIND" = cvae ]; then   # E14 CVAE (cvae/export.py dir), HF only; REASONING_BUDGET forcing as qwen3_vl_vllm
  M=qwen3_vl_cvae; MA="pretrained=$MODEL,attn_implementation=flash_attention_2,max_num_frames=32,extract_answer=True,latent=${CVAE_LATENT:-mean}"
  [ -n "${REASONING_BUDGET:-}" ] && MA="$MA,reasoning_budget=$REASONING_BUDGET"
  export QWEN3VL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)${OPSD_SYSTEM_SUFFIX:-}"
  GEN=(--gen_kwargs max_new_tokens=${MAX_NEW_TOKENS:-1024},temperature=0,top_p=1,do_sample=false)
else
  M=future_l1; MA="pretrained=$MODEL,code_root=$ROOT,attn_implementation=flash_attention_2,max_num_frames=32,extract_answer=True"
  GEN=(--gen_kwargs max_new_tokens=${MAX_NEW_TOKENS:-1024},temperature=0,top_p=1,do_sample=false)  # was 2048 before 2026-09-27
fi
if [ "$BACKEND" = vllm ]; then
  case $KIND in future_l1|cvae) echo "BACKEND=vllm does not support KIND=$KIND"; exit 1;; esac
  M=qwen3_vl_vllm; MA=${MA/attn_implementation=flash_attention_2,/}${VLLM_MA_EXTRA:-}; BATCH=${VLLM_BATCH:-256}
  # REASONING_BUDGET (opsd): cap the reasoning, force "</reason><answer>" on outputs that have no answer by then
  # (reasoning_budget=... is already in MA from the opsd branch above)
  [ -n "${MAX_PIXELS:-}" ] && MA="$MA,max_pixels=$MAX_PIXELS"   # e.g. FutureOmni's official 602112 (768*28*28)
fi

TASKS=${TASKS:-"seedbench_r1_l1:seedbench_r1/l1 seedbench_r1_l2:seedbench_r1/l2 seedbench_r1_l3:seedbench_r1/l3
       tempcompass_multi_choice:tempcompass/multi_choice tempcompass_yes_no:tempcompass/yes_no tempcompass_caption_matching:tempcompass/caption_matching
      
       temporalbench_short_qa:temporalbench/short_qa"}   # override with TASKS="task:out/sub ...". Since 2026-09-27 only TB short_qa (the reported TB metric is short multi-binary;
# long_qa doubled the cost). short_caption needs an LLM judge (empty in our earlier evaluation code too)

for pair in $TASKS; do
  task=${pair%%:*}; sub=${pair#*:}; d=$OUT/$sub
  [ -f $d/results.json ] && { say "$task already done"; continue; }
  for attempt in 1 2; do   # retry once: FFmpeg can abort a rank at random ("Assertion ret >= 0 failed at libavcodec/bitstream.c")
  mkdir -p $d/raw; gpu_ready; say "$task ...$([ $attempt = 2 ] && echo ' (retry)')"
  ( cd $ROOT/eval/lmms-eval && timeout -k 120 ${LAUNCH_TIMEOUT:-5400} accelerate launch --num_processes=$NGPU --main_process_port=$PORT -m lmms_eval \
      --model $M --model_args "$MA" --tasks $task --batch_size $BATCH "${GEN[@]}" \
      --log_samples --log_samples_suffix $task --output_path $d/raw ${LIMIT:+--limit $LIMIT} ) > $LOG/$task.log 2>&1
  res=$(ls -t $d/raw/*/*_results.json 2>/dev/null | head -1); smp=$(ls -t $d/raw/*/*_samples_$task.jsonl 2>/dev/null | head -1)
  if [ -n "$res" ] && [ -n "$smp" ]; then cp $res $d/results.json; cp $smp $d/samples.jsonl; say "$task OK"
  else say "$task FAILED (see logs/$task.log)"; fi
  [ -f $d/results.json ] && break
  done
done
python $ROOT/eval/score_video3.py $OUT | tee -a $LOG/progress.log
say "ALL DONE"
