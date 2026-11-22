#!/bin/bash
# General video understanding: Video-MME, LongVideoBench (val, video) and MVBench (19 subtasks; fine_grained_pose is
# excluded because its NTU RGB+D videos are not redistributed). Task configs and data are those of
# our earlier evaluation code (lmms-eval/lmms_eval/tasks/{videomme,longvideobench,mvbench} point at its dataset/), so the
# numbers are comparable to that project. vLLM backend (lmms-eval model `qwen3_vl_vllm`).
#
#   bash eval/eval_general.sh <TAG> <MODEL_PATH> <KIND>
#     KIND base -> direct-answer prompts and each task's own max_new_tokens
#          opsd -> OPSD system prompt (config/opsd_system_prompt.txt), reason then <answer>, max_new_tokens 1024,
#                  only the <answer> content goes to the task's parser
#          future_l1 -> Future-L1 SFT/RL checkpoints: lmms-eval `future_l1` (HF, latent-aware generation, its own
#                  system prompt), <answer> extracted, max_new_tokens 1024 (same settings as eval_video3.sh)
#
# Output: output/<TAG>/{videomme,longvideobench,mvbench}/{results.json,samples*.jsonl}, output/<TAG>/general_summary.json
set -uo pipefail
# LAUNCH_TIMEOUT: kill a model launch after this many seconds (default 90 min). lmms-eval's NCCL timeout is
# 60000 s, so a rank stuck in the final gather otherwise hangs the job for hours with nvidia-smi showing 100%
# (spinning NCCL kernels); the timeout turns such a hang into a retry.
TAG=${1:?tag}; MODEL=${2:?model path}; KIND=${3:?base|opsd}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
OUT=$OUTPUT_ROOT/$TAG; LOG=$OUT/logs; mkdir -p $LOG
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PORT=${PORT:-$((14000 + RANDOM % 1000))}

# VLLM_ENV: vLLM env for the model (default future-l1-rl; future-l1-q35 = vLLM 0.30 / transformers 5 for Qwen3.5);
# VLLM_MA_EXTRA: appended to the vLLM model_args (Qwen3.5: ",enable_thinking=False" keeps the reason-then-answer protocol)
export PATH=$ENV_ROOT/${VLLM_ENV:-future-l1-rl}/bin:$PATH PYTHONNOUSERSITE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTHONPATH http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HOME=$HF_HOME_EVAL HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENCV_FFMPEG_LOGLEVEL=-8 AV_LOG_LEVEL=quiet
say(){ echo "[general $TAG] $(date -u '+%F %T') $*" | tee -a $LOG/progress.log; }

source $ROOT/utils/gpu_clean.sh   # gpu_ready: kill leftover vLLM processes, wait for empty cards

MA="pretrained=$MODEL${VLLM_MA_EXTRA:-},max_num_frames=32,max_num_seqs=${MAX_NUM_SEQS:-256},preprocess_threads=${PREPROC_THREADS:-28}"
if [ "$KIND" = opsd ]; then
  MA="$MA,extract_answer=True${REASONING_BUDGET:+,reasoning_budget=$REASONING_BUDGET}"   # REASONING_BUDGET: cap + forced answer
  export QWEN3VL_SYSTEM_PROMPT="$(cat $ROOT/config/opsd_system_prompt.txt)${OPSD_SYSTEM_SUFFIX:-}"
  GEN=(--gen_kwargs max_new_tokens=${MAX_NEW_TOKENS:-1024},temperature=0,top_p=1,do_sample=false)
elif [ "$KIND" = base ]; then GEN=()
elif [ "$KIND" = future_l1 ]; then
  export PATH=$ENV_ROOT/future-l1-sft/bin:$PATH FUTURE_L1_ROOT=$ROOT/train/sft FUTURE_L1_CODE_ROOT=$ROOT/train/sft VIDEO_L1_ROOT=$ROOT/train/sft
  M=future_l1; MA="pretrained=$MODEL,code_root=$ROOT,attn_implementation=flash_attention_2,max_num_frames=32,extract_answer=True"
  GEN=(--gen_kwargs max_new_tokens=${MAX_NEW_TOKENS:-1024},temperature=0,top_p=1,do_sample=false)
else echo "KIND must be base, opsd or future_l1"; exit 1; fi
M=${M:-qwen3_vl_vllm}; BS=${VLLM_BATCH:-512}; [ "$M" = future_l1 ] && BS=${BATCH:-16}

MVB="mvbench_action_sequence mvbench_moving_count mvbench_action_prediction mvbench_episodic_reasoning mvbench_action_antonym mvbench_action_count mvbench_scene_transition mvbench_object_shuffle mvbench_object_existence mvbench_unexpected_action mvbench_moving_direction mvbench_state_change mvbench_object_interaction mvbench_character_order mvbench_action_localization mvbench_counterfactual_inference mvbench_fine_grained_action mvbench_moving_attribute mvbench_egocentric_navigation"
# name | lmms-eval tasks (comma list = one launch) ; override with BENCHES="videomme longvideobench mvbench"
declare -A TASKS=([videomme]=videomme [longvideobench]=longvideobench_val_v [mvbench]=${MVB// /,})
BENCHES=${BENCHES:-"mvbench videomme longvideobench"}

for b in $BENCHES; do
  d=$OUT/$b
  [ -f $d/results.json ] && { say "$b already done"; continue; }
  mkdir -p $d/raw; gpu_ready; say "$b ..."
  ( cd $ROOT/eval/lmms-eval && timeout -k 120 ${LAUNCH_TIMEOUT:-5400} accelerate launch --num_processes=$NGPU --main_process_port=$PORT -m lmms_eval \
      --model $M --model_args "$MA" --tasks ${TASKS[$b]} --batch_size $BS "${GEN[@]}" \
      --log_samples --log_samples_suffix $b --output_path $d/raw ${LIMIT:+--limit $LIMIT} ) > $LOG/$b.log 2>&1
  res=$(ls -t $d/raw/*/*_results.json 2>/dev/null | head -1)
  if [ -n "$res" ]; then cp $res $d/results.json; cp $(dirname $res)/*_samples_*.jsonl $d/ 2>/dev/null; say "$b OK"
  else say "$b FAILED (see logs/$b.log)"; fi
done

python3 - "$OUT" <<'EOF' | tee -a $LOG/progress.log
import json, os, sys
out = sys.argv[1]; s = {"tag": os.path.basename(out)}
def res(b):
    f = f"{out}/{b}/results.json"
    return json.load(open(f))["results"] if os.path.exists(f) else {}
def metric(r, key):
    return next((v for k, v in r.items() if k.startswith(key) and "stderr" not in k), None)
r = res("videomme")
if r: s["VideoMME"] = round(metric(r["videomme"], "videomme_perception_score"), 2)
r = res("longvideobench")
if r: s["LongVideoBench"] = round(100 * metric(r["longvideobench_val_v"], "lvb_acc"), 2)
r = res("mvbench")
if r:
    sub = {k.replace("mvbench_", ""): round(metric(v, "mvbench_accuracy"), 2) for k, v in r.items() if k.startswith("mvbench_")}
    s["MVBench"] = {"mean": round(sum(sub.values()) / len(sub), 2), "n_subtasks": len(sub), **sub}
json.dump(s, open(f"{out}/general_summary.json", "w"), indent=1)
print(json.dumps({k: (v["mean"] if isinstance(v, dict) else v) for k, v in s.items()}, ensure_ascii=False))
EOF
say "ALL DONE"
