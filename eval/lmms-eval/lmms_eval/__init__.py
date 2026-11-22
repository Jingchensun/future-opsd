import os as _os

# Dataset paths in the task configs are written as ${DATA_ROOT}/... (see config/paths.sh of the Future-OPSD repo);
# default: <repo>/dataset, with this package at <repo>/eval/lmms-eval/lmms_eval.
_os.environ.setdefault("DATA_ROOT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), "dataset"))
