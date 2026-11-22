"""Download every third-party dataset from its public source into DATA_ROOT (layout: data_tools/layout.py) and build the
files the code reads (lmms-eval `datasets` folders, Future-L1 OPSD json). Run on a machine with internet; each step is
resumable (finished downloads are skipped by huggingface_hub, extracted archives are marked with .done files).

    python data_tools/download/public.py <step> [<step> ...]        # or `all`
    python data_tools/download/public.py verify                     # check that every referenced video / image exists

steps (sizes are what ends up on disk):
  egoplan_train   EgoPlan-IT = SEED-Bench-R1 training_50k (+ cot_6k): annotations, images, 391 EpicKitchens clip
                  tarballs (~400 GB extracted), EPIC-KITCHENS-100 annotation csv               -> egoplan/train/
  egoplan_test    SEED-Bench-R1 validation L1/L2/L3: annotations, images, the 4,143 clips the splits use (from the
                  Ego4D + EpicKitchens tarballs), ds_l1..ds_l3                                -> egoplan/test/
  future_l1_train Future-L1-50K (Eurayka/Future-L1-50K): json + videos streamed out of the parquet shards (~256 GB),
                  Future-L1-50K.videos.json and Future-L1-50K.opsd.json                       -> future_l1/train/
  future_l1_test  TwiFF-Bench (Liu-Junhua/TwiFF-Bench, 4 parquet)                             -> future_l1/test/
  tempcompass     lmms-lab/TempCompass: videos + ds_{multi-choice,yes_no,caption_matching,captioning}
  temporalbench   microsoft/TemporalBench: short / long videos + ds (splits test_short_qa, test_long_qa, test_short_caption)
  videomme        lmms-lab/Video-MME: 900 videos (20 zip chunks) + ds
  mvbench         OpenGVLab/MVBench (gated: accept the terms on the HF page first, then `hf auth login`): videos + ds/<task>;
                  SSv2 clips are re-encoded VP9 .webm -> H.264 .mp4 (decord cannot decode VP9). fine_grained_pose needs
                  the NTU RGB+D videos, which are not redistributed (the paper reports 19 of the 20 MVBench tasks).
Derived training data (EgoPlan-IT jsonl + future frames, data-scaling subsets, rejection-sampling SFT data) is built
with the scripts in data_tools/ (see README.md).
"""
import glob, json, os, shutil, subprocess, sys, tarfile, zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout import DATA_ROOT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = "TencentARC/SEED-Bench-R1"
EK_ANNOT = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/"


def D(*p):
    path = os.path.join(DATA_ROOT, *p); os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


def hf(repo, local, patterns=None, workers=8):
    from huggingface_hub import snapshot_download
    for k in range(5):
        try:
            return snapshot_download(repo, repo_type="dataset", local_dir=local, allow_patterns=patterns, max_workers=workers)
        except Exception as e:  # transient network errors: retry, finished files are kept
            print(f"retry {k} {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    raise RuntimeError(f"download failed: {repo}")


def once(marker, fn):
    """Run fn() unless <marker>.done exists (archives are extracted once)."""
    if os.path.exists(marker + ".done"):
        return
    fn(); open(marker + ".done", "w").close()


def untar(path, dest, keep=False):
    once(path, lambda: (os.makedirs(dest, exist_ok=True), tarfile.open(path).extractall(dest)))
    if not keep and os.path.exists(path):
        os.remove(path)


def unzip(path, dest, rename=None, keep=False):
    """Extract a zip; rename(member) -> path relative to dest (None = skip)."""
    def run():
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                rel = rename(m.filename) if rename else m.filename
                if rel is None:
                    continue
                out = os.path.join(dest, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(m) as src, open(out + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(out + ".part", out)
    once(path, run)
    if not keep and os.path.exists(path):
        os.remove(path)


def _plain(name):  # keep the archive's own paths, skip macOS metadata
    return None if "__MACOSX" in name or os.path.basename(name).startswith("._") else name


def save_ds(rows_by_split, out):
    from datasets import Dataset, DatasetDict
    shutil.rmtree(out, ignore_errors=True)
    DatasetDict({s: Dataset.from_list(r) for s, r in rows_by_split.items()}).save_to_disk(out)
    print("built", out, {s: len(r) for s, r in rows_by_split.items()}, flush=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def read_parquet(p):
    import pyarrow.parquet as pq
    return pq.read_table(p).to_pylist()


# ---------------------------------------------------------------- EgoPlan-IT / SEED-Bench-R1
def egoplan_train():
    T = D("egoplan", "train")
    hf(SEED, T, ["annotations/training_*", "images.tar.gz", "videos/EpicKitchens/*", "README.md"])
    untar(f"{T}/images.tar.gz", T)
    tars = sorted(glob.glob(f"{T}/videos/EpicKitchens/batch_*.tar.gz"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: untar(p, f"{T}/videos/EpicKitchens"), tars))
    for f in ["EPIC_100_train.csv", "EPIC_100_validation.csv", "EPIC_100_video_info.csv", "EPIC_100_test_timestamps.csv"]:
        if not os.path.exists(f"{T}/ek100/{f}"):
            subprocess.run(["curl", "-sfL", "--retry", "5", "-o", D("egoplan", "train", "ek100", f), EK_ANNOT + f], check=True)
    print("egoplan_train done; next: python data_tools/make_egoplan_train.py (training files + future frames)")


def build_seed_ds(T):
    for L in (1, 2, 3):  # sample_id as string, no clip -> ""; L1/L2 lack domain / scenario / task_start_frame -> "", "", 0
        rows = []
        for r in read_jsonl(f"{T}/annotations/validation_L{L}.jsonl"):
            r["sample_id"] = str(r["sample_id"])
            if r.get("video_basename") is None:  # tasks with no progress yet have no clip
                r["video_basename"] = ""
            for k, v in (("domain", ""), ("scenario", ""), ("task_start_frame", 0)):
                r.setdefault(k, v)
            rows.append(r)
        save_ds({"test": rows}, f"{T}/ds_l{L}")


def egoplan_test():
    from huggingface_hub import HfApi, hf_hub_download
    T = D("egoplan", "test")
    hf(SEED, T, ["annotations/validation_*", "images.tar.gz", "README.md"])
    untar(f"{T}/images.tar.gz", T)
    need = {r["video_basename"] for L in (1, 2, 3) for r in read_jsonl(f"{T}/annotations/validation_L{L}.jsonl")
            if r.get("video_basename")}
    os.makedirs(f"{T}/videos", exist_ok=True)
    train_clips = os.path.join(DATA_ROOT, "egoplan", "train", "videos", "EpicKitchens")
    for b in sorted(need - set(os.listdir(f"{T}/videos"))):  # EpicKitchens clips already extracted by egoplan_train
        if os.path.exists(f"{train_clips}/{b}"):
            shutil.copy(f"{train_clips}/{b}", f"{T}/videos/{b}")
    missing = need - set(os.listdir(f"{T}/videos"))
    files = [s.rfilename for s in HfApi().dataset_info(SEED).siblings if s.rfilename.startswith("videos/")]
    if not any(not b.startswith("P") for b in missing):  # only EpicKitchens clips missing -> skip Ego4D tarballs
        files = [f for f in files if "EpicKitchens" in f]
    if missing:
        def work(fn):
            marker = f"{T}/.tar_done/" + fn.replace("/", "__")
            if os.path.exists(marker):
                return
            p = hf_hub_download(SEED, fn, repo_type="dataset", local_dir=f"{T}/.tmp")
            with tarfile.open(p, "r:gz") as t:
                for m in t:
                    if m.isfile() and os.path.basename(m.name) in need:
                        m.name = os.path.basename(m.name); t.extract(m, f"{T}/videos")
            os.remove(p); os.makedirs(os.path.dirname(marker), exist_ok=True); open(marker, "w").close()
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(work, files))
    print("SEED-Bench-R1 clips missing:", len(need - set(os.listdir(f"{T}/videos"))), "of", len(need))
    build_seed_ds(T)




# ---------------------------------------------------------------- Future-L1
def future_l1_train():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    T = D("future_l1", "train"); repo = "Eurayka/Future-L1-50K"
    hf(repo, T, ["Future-L1-50K.json", "README.md"])
    os.makedirs(f"{T}/videos", exist_ok=True); os.makedirs(f"{T}/_fetch_done", exist_ok=True)

    def fetch(shard):  # write the clips embedded in one parquet shard, then drop the shard (peak disk ~ final size)
        ok = f"{T}/_fetch_done/{os.path.basename(shard)}.ok"
        if os.path.exists(ok):
            return
        local = hf_hub_download(repo, shard, repo_type="dataset", local_dir=f"{T}/_shards")
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            for v in pf.read_row_group(rg, columns=["video"]).column("video").to_pylist():
                dst = f"{T}/videos/{v['path']}"
                if not os.path.exists(dst):
                    with open(dst + ".tmp", "wb") as f:
                        f.write(v["bytes"])
                    os.replace(dst + ".tmp", dst)
        os.remove(local); open(ok, "w").close()
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, shards))
    data = json.load(open(f"{T}/Future-L1-50K.json"))
    # SFT reads videos/<file> relative to the json; OPSD / GRPO (verl) read absolute paths and do not take meta_data
    json.dump([{**x, "video": "videos/" + x["video"]} for x in data], open(f"{T}/Future-L1-50K.videos.json", "w"))
    json.dump([{k: (f"{T}/videos/{v}" if k == "video" else v) for k, v in x.items() if k != "meta_data"} for x in data],
              open(f"{T}/Future-L1-50K.opsd.json", "w"))
    print("Future-L1-50K: videos missing", sum(not os.path.exists(f"{T}/videos/{x['video']}") for x in data))


def future_l1_test():
    hf("Liu-Junhua/TwiFF-Bench", D("future_l1", "test"), ["test-*.parquet", "README.md"])


# ---------------------------------------------------------------- general video benchmarks
def tempcompass():
    G = D("general", "tempcompass")
    hf("lmms-lab/TempCompass", G)
    unzip(f"{G}/tempcompass_videos.zip", G, rename=_plain)  # -> videos/*.mp4
    for c in ["multi-choice", "yes_no", "caption_matching", "captioning"]:
        os.makedirs(f"{G}/anno", exist_ok=True)
        if os.path.isdir(f"{G}/{c}") and not os.path.exists(f"{G}/anno/{c}"):
            shutil.move(f"{G}/{c}", f"{G}/anno/{c}")
        save_ds({"test": read_parquet(glob.glob(f"{G}/anno/{c}/test-*.parquet")[0])}, f"{G}/ds_{c}")


def temporalbench():
    G = D("general", "temporalbench")
    hf("microsoft/TemporalBench", G)
    for z in sorted(glob.glob(f"{G}/*.zip")):  # -> short_video/<source>/*.mp4, long_video/<source>/*.mp4
        unzip(z, G, rename=_plain)
    save_ds({s: read_parquet(glob.glob(f"{G}/data/{s}-*.parquet")[0]) for s in
             ["test_short_qa", "test_long_qa", "test_short_caption"]}, f"{G}/ds")


def videomme():
    G = D("general", "videomme")
    hf("lmms-lab/Video-MME", G, ["videomme/*.parquet", "videos_chunked_*.zip", "README.md"])
    for z in sorted(glob.glob(f"{G}/videos_chunked_*.zip")):
        unzip(z, G, rename=_plain)  # -> data/<videoID>.mp4
    save_ds({"test": read_parquet(glob.glob(f"{G}/videomme/test-*.parquet")[0])}, f"{G}/ds")


def mvbench():
    G = D("general", "mvbench")
    hf("OpenGVLab/MVBench", G, ["json/*", "video/*", "README.md"])
    for z in sorted(glob.glob(f"{G}/video/*.zip")):
        unzip(z, G, rename=_plain)  # -> clevrer/ perception/ ssv2_video/ ... (the folders of tasks/mvbench/utils.py)
    if not os.path.exists(f"{G}/ssv2_video_mp4"):  # the task config reads action_antonym from ssv2_video_mp4/
        os.symlink("ssv2_video", f"{G}/ssv2_video_mp4")
    for w in sorted(glob.glob(f"{G}/ssv2_video/*.webm")):  # VP9 -> H.264 next to the .webm (picked up by tasks/mvbench/utils.py)
        if not os.path.exists(w[:-5] + ".mp4"):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", w, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", w[:-5] + ".mp4"], check=True)
    for j in sorted(glob.glob(f"{G}/json/*.json")):
        save_ds({"train": json.load(open(j))}, f"{G}/ds/{os.path.basename(j)[:-5]}")


STEPS = {f.__name__: f for f in [egoplan_train, egoplan_test, future_l1_train, future_l1_test,
                                  tempcompass, temporalbench, videomme, mvbench]}


def verify():
    """Every file the evaluation / training data points to exists."""
    from datasets import load_from_disk
    bad = {}
    def chk(name, paths):
        miss = [p for p in paths if not os.path.exists(p)]; bad[name] = len(miss)
        print(f"{name}: {len(paths) - len(miss)}/{len(paths)} present", (miss[:3] if miss else ""), flush=True)
    T = os.path.join(DATA_ROOT, "egoplan", "test")
    for L in (1, 2, 3):
        rows = load_from_disk(f"{T}/ds_l{L}")["test"]
        chk(f"seed_l{L} videos", [f"{T}/videos/{r['video_basename']}" for r in rows if r["video_basename"]])
        chk(f"seed_l{L} images", [f"{T}/images/{r['video_source']}/{r['current_observation_basename']}" for r in rows])
    G = os.path.join(DATA_ROOT, "general")
    chk("videomme", sorted({f"{G}/videomme/data/{r['videoID']}.mp4" for r in load_from_disk(f"{G}/videomme/ds")["test"]}))
    chk("tempcompass", sorted({f"{G}/tempcompass/videos/{r['video_id']}.mp4" for r in load_from_disk(f"{G}/tempcompass/ds_multi-choice")["test"]}))
    chk("temporalbench", sorted({f"{G}/temporalbench/{r['video_name']}" for r in load_from_disk(f"{G}/temporalbench/ds")["test_short_qa"]}))
    T = os.path.join(DATA_ROOT, "future_l1", "train")
    if os.path.exists(f"{T}/Future-L1-50K.opsd.json"):
        chk("future_l1 videos", [x["video"] for x in json.load(open(f"{T}/Future-L1-50K.opsd.json"))])
    return bad


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args == ["verify"]:
        verify(); sys.exit(0)
    for s in (list(STEPS) if args == ["all"] else args):
        print("=====", s, flush=True); STEPS[s]()
