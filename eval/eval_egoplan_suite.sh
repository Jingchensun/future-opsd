#!/bin/bash
# E18/E19 evaluation set (OPSD protocol: vLLM, OPSD system prompt, reasoning budget 512): FutureBench, TwiFF (+ Qwen3.6-27B judge),
# SEED-Bench-R1 L1/L2/L3, TempCompass, Video-MME and MVBench (TemporalBench: run eval_video3.sh with its task separately).
#   bash eval/eval_egoplan_suite.sh <TAG> <MODEL_DIR>
# Output: output/<TAG>/{summary.json, video3_summary.json, general_summary.json}
set -uo pipefail
TAG=${1:?tag}; MODEL=${2:?model dir}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
export REASONING_BUDGET=${REASONING_BUDGET:-512} VLLM_BATCH=${VLLM_BATCH:-128} PREPROC_THREADS=${PREPROC_THREADS:-16}
BACKEND=vllm bash $ROOT/eval/eval_model.sh $TAG $MODEL opsd ${EGO_STAGES:-fb,twiff,judge}   # fb = FutureBench (added for E19)
TASKS="seedbench_r1_l1:seedbench_r1/l1 seedbench_r1_l2:seedbench_r1/l2 seedbench_r1_l3:seedbench_r1/l3
       tempcompass_multi_choice:tempcompass/multi_choice tempcompass_yes_no:tempcompass/yes_no
       tempcompass_caption_matching:tempcompass/caption_matching" BACKEND=vllm bash $ROOT/eval/eval_video3.sh $TAG $MODEL opsd
BENCHES="mvbench videomme" bash $ROOT/eval/eval_general.sh $TAG $MODEL opsd
echo "[egoplan-suite $TAG] scores:"
for f in summary.json video3_summary.json general_summary.json; do [ -f $OUTPUT_ROOT/$TAG/$f ] && head -c 600 $OUTPUT_ROOT/$TAG/$f && echo; done
