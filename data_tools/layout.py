"""Dataset layout under DATA_ROOT (default <repo>/dataset) and the placeholder used in shared data files.

    egoplan/train/       EgoPlan-IT (SEED-Bench-R1 training_50k): annotations/ images/ videos/ ek100/ (EPIC-KITCHENS-100 csv)
                         + built files: egoplan_train*.jsonl, egoplan_sft_cot6k.json, future_frames/, scale/, rft/
    egoplan/test/        SEED-Bench-R1 validation L1/L2/L3: annotations/ images/ videos/ ds_l1..ds_l3 (load_from_disk)
    egoplan/ek100_raw/   EPIC-KITCHENS-100 source videos (only for the 50K pool of the data-scaling study)
    future_l1/train/     Future-L1-50K: Future-L1-50K.json, .opsd.json, .videos.json, videos/
    future_l1/test/      TwiFF-Bench parquet
    general/             tempcompass/ temporalbench/ videomme/ mvbench/   (videos + ds* built for lmms-eval)
    derived/             captions/ (z = text), multi_future/ (Gemini futures)
    extra/               data of exploratory runs that are not in the paper tables
                         (v1_33k, v1_33k_rl_2k.json, v1_33k_transcoded, futurebench, futureomni, egoplan_bench2, longvideobench)

Files that store paths (jsonl / json) use absolute paths on the machine they were built on. The HF backup stores them
with the placeholder `{DATA_ROOT}/`; data_tools/localize_paths.py turns the placeholder into the local DATA_ROOT.
"""
import os

PLACEHOLDER = "{DATA_ROOT}/"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "dataset"))

# Absolute path prefixes of an older private layout -> layout path (used only when building the data backup).
LEGACY = []


def to_placeholder(text: str) -> str:
    """Absolute legacy / local paths -> {DATA_ROOT}/<layout path>."""
    for old, new in LEGACY:
        text = text.replace(old, PLACEHOLDER + new)
    return text.replace(DATA_ROOT.rstrip("/") + "/", PLACEHOLDER)


def to_local(text: str, data_root: str = DATA_ROOT) -> str:
    """{DATA_ROOT}/ and legacy absolute paths -> paths under data_root."""
    return to_placeholder(text).replace(PLACEHOLDER, data_root.rstrip("/") + "/")
