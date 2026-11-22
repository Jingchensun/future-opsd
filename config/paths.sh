# Paths of a Future-OPSD checkout. Every script sources this file; override any of them in the environment.
#   ROOT         the repository
#   DATA_ROOT    datasets (layout: data_tools/layout.py)      default $ROOT/dataset
#   CKPT_ROOT    base models (Qwen3-VL-8B-Instruct, ...) and training outputs   default $ROOT/ckpt
#   OUTPUT_ROOT  evaluation results / logs                    default $ROOT/output
#   ENV_ROOT     the virtualenvs of env/install_env.sh        default $ROOT/envs
#   HF_HOME_EVAL HF cache used offline by the evaluations     default $TMPDIR/hf_home_futurel1 (any writable local dir)
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export ROOT
export DATA_ROOT=${DATA_ROOT:-$ROOT/dataset}
export CKPT_ROOT=${CKPT_ROOT:-$ROOT/ckpt}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output}
export ENV_ROOT=${ENV_ROOT:-$ROOT/envs}
export HF_HOME_EVAL=${HF_HOME_EVAL:-${TMPDIR:-/tmp}/hf_home_futurel1}
