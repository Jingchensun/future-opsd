# source from the eval scripts; call gpu_ready before every model launch.
# A crashed launch (or a judge server's tensor-parallel worker) can leave a vLLM process that keeps ~150 GB on one card;
# every later launch then fails on that rank. Kill leftovers and wait until the cards are (almost) empty.
gpu_ready(){
  pkill -9 -f "[V]LLM::" 2>/dev/null                       # EngineCore / Worker_TP* process titles
  pkill -9 -f "[l]mms_eval --model" 2>/dev/null
  pkill -9 -f "[v]llm serve" 2>/dev/null
  local used=0
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 20000 ] && return 0; sleep 10
  done
  echo "[gpu_ready] warning: a GPU still holds ${used} MiB after 10 min" >&2; return 1
}
