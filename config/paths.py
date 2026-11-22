"""Python side of config/paths.sh (same environment variables and defaults)."""
import os

ROOT = os.environ.get("ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.join(ROOT, "dataset")
CKPT_ROOT = os.environ.get("CKPT_ROOT") or os.path.join(ROOT, "ckpt")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT") or os.path.join(ROOT, "output")
ENV_ROOT = os.environ.get("ENV_ROOT") or os.path.join(ROOT, "envs")
