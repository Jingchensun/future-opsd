#!/bin/bash
# Recreate an environment used for the paper from its exact package lock, env/lock/<env>.txt (a `pip freeze --all` of
# the env the runs used).
#
#   bash env/install_env.sh <env> [<env> ...]      env: future-l1-rl-vllm011 future-l1-rl future-l1-q35 judge-vllm future-l1-sft
#   bash env/install_env.sh all
#
# What it does, per env:
#   1. creates $ENV_ROOT/<env> as a Python 3.11 venv (refuses to overwrite an existing one unless FORCE=1);
#   2. installs every pinned package with --no-deps, i.e. exactly the frozen set. All of them come from PyPI, torch
#      included: torch 2.8 / 2.9 resolve to the CUDA 12.8 builds and torch 2.13 to the CUDA 13.0 builds, as in the lock;
#   3. flash-attn 2.8.3: future-l1-rl-vllm011 gets the official prebuilt wheel (pinned in its lock); future-l1-rl and
#      future-l1-sft use torch 2.9, for which no official wheel exists, so it is compiled once (about 30-60 min) into
#      $ENV_ROOT/wheels/ and reused by the other env;
#   4. puts train/rl on the path of the Qwen3-VL envs (as the original envs did) and installs eval/lmms-eval in editable
#      mode where the lock lists it;
#   5. imports the key packages and prints their versions, then runs `pip check` (informational).
#
# Settings (environment variables):
#   PYTHON    interpreter for the venv (default python3.11)
#   FA_ARCHS  CUDA architectures for the flash-attn build, ";"-separated: 80 = A100, 90 = H100/H200, 100 = B200.
#             Default: the compute capability of the local GPUs (nvidia-smi). Required on a host without GPUs.
#   MAX_JOBS  parallel nvcc jobs for the flash-attn build (default 4; each takes ~5 GB of host memory)
#   FORCE=1   delete and rebuild an existing env
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
PYTHON=${PYTHON:-python3.11}
ALL=(future-l1-rl-vllm011 future-l1-rl future-l1-q35 judge-vllm future-l1-sft)
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 1; }
[ "$1" = all ] && set -- "${ALL[@]}"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1; unset PYTHONPATH
mkdir -p $ENV_ROOT/logs $ENV_ROOT/wheels
say(){ echo "[install $1] $(date -u +%T) ${*:2}"; }

"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' 2>/dev/null \
  || { echo "need Python 3.11 (set PYTHON=/path/to/python3.11)"; exit 1; }

fa_archs(){
  if [ -n "${FA_ARCHS:-}" ]; then echo "$FA_ARCHS"; return; fi
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d . | sort -u | paste -sd';' - | grep . && return
  echo "no GPU found: set FA_ARCHS (e.g. FA_ARCHS=90 for H100, 100 for B200)" >&2; return 1
}

build_flash_attn(){  # $1 = env dir (torch 2.9.1); builds into $ENV_ROOT/wheels once per arch set
  local E=$1 archs tag whl
  archs=$(fa_archs); tag=$(echo "$archs" | tr ';' '_')
  whl=$(ls $ENV_ROOT/wheels/sm$tag/flash_attn-2.8.3-*.whl 2>/dev/null | head -1 || true)
  if [ -z "$whl" ]; then
    command -v nvcc >/dev/null || { echo "flash-attn needs the CUDA toolkit (nvcc >= 12.8) to compile"; return 1; }
    say "$(basename $E)" "compiling flash-attn 2.8.3 for sm_${archs} with MAX_JOBS=${MAX_JOBS:-4} (log: $ENV_ROOT/logs/flash_attn_sm$tag.log)"
    mkdir -p $ENV_ROOT/wheels/sm$tag
    FLASH_ATTN_CUDA_ARCHS="$archs" MAX_JOBS=${MAX_JOBS:-4} NVCC_THREADS=1 \
      $E/bin/pip wheel --no-build-isolation --no-deps --no-cache-dir flash-attn==2.8.3 -w $ENV_ROOT/wheels/sm$tag \
      > $ENV_ROOT/logs/flash_attn_sm$tag.log 2>&1 || { echo "flash-attn build failed, see the log"; return 1; }
    whl=$(ls $ENV_ROOT/wheels/sm$tag/flash_attn-2.8.3-*.whl | head -1)
  fi
  $E/bin/pip install --no-deps "$whl"
}

install_one(){
  local name=$1 E=$ENV_ROOT/$1 lock=$ROOT/env/lock/$1.txt
  [ -f $lock ] || { echo "unknown env '$name' (choose from: ${ALL[*]})"; return 1; }
  if [ -e $E ]; then
    [ "${FORCE:-0}" = 1 ] || { echo "$E exists; set FORCE=1 to rebuild it"; return 1; }
    rm -rf $E
  fi
  say $name "creating venv ($("$PYTHON" -V)) at $E"
  # stdlib venv needs ensurepip (Debian/Ubuntu: apt install python3.11-venv); otherwise use virtualenv, bootstrapped
  # into $ENV_ROOT/.bootstrap with pip if it is not installed (no root needed, the system Python is left untouched)
  if ! "$PYTHON" -m venv $E > /dev/null 2>&1; then
    rm -rf $E
    local BS=$ENV_ROOT/.bootstrap
    if ! "$PYTHON" -m virtualenv --version > /dev/null 2>&1 && ! PYTHONPATH=$BS "$PYTHON" -m virtualenv --version > /dev/null 2>&1; then
      say $name "python3.11-venv is missing: bootstrapping virtualenv into $BS"
      "$PYTHON" -m pip install -q --target $BS "virtualenv==20.29.3" || {
        echo "cannot create a venv: install python3.11-venv (apt) or pip for $PYTHON"; return 1; }
    fi
    PYTHONPATH=$BS "$PYTHON" -m virtualenv -q $E || { echo "virtualenv failed to create $E"; return 1; }
  fi
  $E/bin/pip install -q --upgrade pip
  say $name "installing $(grep -cvE '^\s*(#|$)' $lock) pinned packages (--no-deps; log: $ENV_ROOT/logs/$name.log)"
  ( cd $ROOT && $E/bin/pip install --no-deps -r $lock ) > $ENV_ROOT/logs/$name.log 2>&1 \
    || { echo "pip install failed for $name, see $ENV_ROOT/logs/$name.log"; return 1; }
  case $name in
    future-l1-rl|future-l1-sft) build_flash_attn $E ;;
  esac
  case $name in
    future-l1-rl|future-l1-rl-vllm011) echo "$ROOT/train/rl" > $E/lib/python3.11/site-packages/future_opsd_trainer.pth ;;
  esac
  local mods
  case $name in
    future-l1-rl-vllm011) mods="torch vllm transformers flash_attn ray verl qwen_vl_utils decord" ;;
    future-l1-rl)         mods="torch vllm transformers flash_attn lmms_eval decord" ;;
    future-l1-q35)        mods="torch vllm transformers deepspeed ray lmms_eval decord" ;;
    judge-vllm)           mods="torch vllm transformers" ;;
    future-l1-sft)        mods="torch transformers deepspeed flash_attn lmms_eval qwen_vl_utils decord peft" ;;
  esac
  say $name "checking imports"
  ( cd /tmp && $E/bin/python - $mods <<'PY' 2>&1 | grep -E "^\[check\]" )
import importlib, sys
import torch
print(f"[check] torch {torch.__version__} (CUDA {torch.version.cuda})")
bad = 0
for m in sys.argv[1:]:
    if m == "torch": continue
    try:
        mod = importlib.import_module(m); print(f"[check]   {m:14s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        bad += 1; print(f"[check]   {m:14s} FAILED: {type(e).__name__}: {str(e)[:150]}")
print("[check] OK" if not bad else f"[check] {bad} import(s) failed")
PY
  $E/bin/pip check > $ENV_ROOT/logs/$name.pipcheck 2>&1 || true
  say $name "done (pip check report: $ENV_ROOT/logs/$name.pipcheck)"
}

for e in "$@"; do install_one $e; done
