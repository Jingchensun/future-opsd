"""Build EgoPlan-IT (SEED-Bench-R1 training_50k) training files for SFT / GRPO / OPSD.

Inputs (dataset/EgoPlan-IT/): annotations/training_50k.jsonl, annotations/training_with_cot_6k.jsonl,
videos/EpicKitchens/*.mp4 (task start -> current observation; original frames re-encoded 1:1 at 30 fps),
images/EpicKitchens/*.png, ek100/EPIC_100_{train,validation,video_info}.csv. With --raw, an answer action that no EgoPlan clip
covers (6,975 of the 50,269 questions) takes its future frames from the EK-100 source video instead
(EPIC-KITCHENS-100 source videos under $DATA_ROOT/egoplan/ek100_raw/videos/<P>/<video_id>.MP4; not needed for Tables 1-2), resized to the clip size 456x256. Source frame index =
EK frame number - 1 (the annotation timeline); the EgoPlan clips sit 22-28 frames (~0.4 s at 60 fps) earlier than the source
at the same frame number (checked on P04_31 / P01_15), so the two kinds of future frames differ by about that much.

Outputs:
  egoplan_train.jsonl   one row per training question that has >= 1 future frame (used by GRPO and every OPSD arm):
      problem   = the SEED-Bench-R1 evaluation prompt: "\n{question}\nA. ..\nB. ..\nC. ..\nD. ..\n{post_prompt}"
                  (options = answer + 3 negatives sampled with a per-sample seed, shuffled)
      answer    = the option letter
      videos / images = [clip path] (empty when the task has no progress yet) / [observation image path]
      opsd_future_frames = up to 3 frames inside the answer action (EK-100 narration timing), taken from a later clip
                           of the same video, saved to future_frames/<sample_id>_<k>.jpg
      opsd_future_text   = the answer action and the next ones, from the EK-100 narrations
      opsd_answer_text   = "<letter>. <answer>"
  egoplan_sft_cot6k.json   training_with_cot_6k in the SFT "messages + videos + images" format, target
                           "<reason>\\n{think}\\n</reason>\\n<answer>{letter}</answer>" (the OPSD / evaluation format)

    python data_tools/make_egoplan_train.py [--workers 32] [--limit N]
    python data_tools/make_egoplan_train.py --raw --no_sft --workers 8 --out dataset/EgoPlan-IT/egoplan_train_50k.jsonl
"""
import argparse, csv, collections, json, os, random, re
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.join(ROOT, "dataset")
D = os.path.join(DATA_ROOT, "egoplan", "train")
POST = "Please provide only the single option letter within the <answer> </answer> tags."  # seedbench_r1 yaml post_prompt
N_FUTURE_TEXT = 3
FRAC = (0.25, 0.5, 0.75)  # future frames at these fractions of the answer action

_CLIPS = None
RAW = os.path.join(DATA_ROOT, "egoplan", "ek100_raw", "videos")
CLIP_SIZE = (456, 256)
_USE_RAW = False


def obs_frame(r):
    return int(re.search(r"-(\d+)\.\w+$", r["current_observation_basename"]).group(1))


def load_clips():
    clips = collections.defaultdict(list)
    for n in os.listdir(os.path.join(D, "videos", "EpicKitchens")):
        m = re.match(r"(.+)-(\d+)_(\d+)\.mp4$", n)
        if m:
            clips[m.group(1)].append((int(m.group(2)), int(m.group(3)), n))
    return clips


def plan(r, nar, fps):
    """-> (answer action (start, stop), future text, [(clip name, clip frame index, target frame)]) or None."""
    v, e = r["video_id"], obs_frame(r)
    f = fps.get(v, 50.0)
    after = [x for x in nar.get(v, []) if x[0] >= e - int(0.5 * f)]
    hit = [x for x in after if x[2] == r["answer"] and x[0] <= e + int(10 * f)]
    if not hit:
        return None
    a0, a1, _ = hit[0]
    nxt = []
    for x in after:  # the answer action and the next distinct ones (EK-100 often repeats a narration back to back)
        if x[0] >= a0 and (not nxt or x[2] != nxt[-1]):
            nxt.append(x[2])
    nxt = nxt[:N_FUTURE_TEXT]
    text = "The next actions in this video are: " + ", then ".join(nxt) + "."
    picks = []
    for fr in FRAC:
        t = a0 + int(fr * max(a1 - a0, 1))
        cov = [c for c in _CLIPS.get(v, []) if c[0] <= t <= c[1]]
        if cov:
            s, _, n = min(cov, key=lambda c: c[1] - c[0])  # the shortest covering clip decodes fastest
            picks.append((n, t - s, t))
        elif _USE_RAW and os.path.exists(os.path.join(RAW, v.split("_")[0], v + ".MP4")):
            picks.append(("RAW:" + os.path.join(RAW, v.split("_")[0], v + ".MP4"), t - 1, t))  # EK frame numbers start at 1
    return (a0, a1), text, picks


def extract(job):
    sid, picks = job
    import decord
    from PIL import Image
    out = []
    by_clip = collections.defaultdict(list)
    for n, idx, t in picks:
        by_clip[n].append((idx, t))
    for n, items in by_clip.items():
        paths = [(t, os.path.join(D, "future_frames", f"{sid}_{t}.jpg")) for _, t in items]
        if all(os.path.exists(p) for _, p in paths):
            out += paths
            continue
        try:
            raw = n.startswith("RAW:")
            vr = decord.VideoReader(n[4:] if raw else os.path.join(D, "videos", "EpicKitchens", n))
            for idx, t in items:
                idx = min(idx, len(vr) - 1)
                p = os.path.join(D, "future_frames", f"{sid}_{t}.jpg")
                if not os.path.exists(p):
                    im = Image.fromarray(vr[idx].asnumpy())
                    (im.resize(CLIP_SIZE, Image.BICUBIC) if raw else im).save(p, quality=92)
                out.append((t, p))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {sid} {n}: {exc}", flush=True)
    return sid, [p for _, p in sorted(out)]


def extract_raw(job):
    """one EK-100 source video -> all the future frames it supplies (opening a 3-10 GB source video takes 40-90 s,
    so it is opened once per video, not once per question); frames are decoded in index order"""
    path, items = job
    import decord
    from PIL import Image
    done = 0
    try:
        vr = decord.VideoReader(path)
        for idx, p in sorted(items):  # one frame at a time: get_batch on 1080p sources OOM-killed a 32 GB host
            Image.fromarray(vr[min(idx, len(vr) - 1)].asnumpy()).resize(CLIP_SIZE, Image.BICUBIC).save(p, quality=92)
            done += 1
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {path}: {exc}", flush=True)
    return path, done, len(items)


def main():
    global _CLIPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(D, "egoplan_train.jsonl"))
    ap.add_argument("--no_sft", action="store_true")
    ap.add_argument("--raw", action="store_true", help="fall back to the EK-100 source videos for uncovered answer actions")
    args = ap.parse_args()
    global _USE_RAW
    _USE_RAW = args.raw
    fps = {r["video_id"]: float(r["fps"]) for r in csv.DictReader(open(os.path.join(D, "ek100", "EPIC_100_video_info.csv")))}
    nar = collections.defaultdict(list)
    for name in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
        for r in csv.DictReader(open(os.path.join(D, "ek100", name))):
            nar[r["video_id"]].append((int(r["start_frame"]), int(r["stop_frame"]), r["narration"]))
    for v in nar:
        nar[v].sort()
    _CLIPS = load_clips()
    rows = [json.loads(l) for l in open(os.path.join(D, "annotations", "training_50k.jsonl"))]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(os.path.join(D, "future_frames"), exist_ok=True)

    plans, stats = {}, collections.Counter()
    for r in rows:
        p = plan(r, nar, fps)
        if p is None:
            stats["no answer narration"] += 1
        elif not p[2]:
            stats["no clip covers the answer action"] += 1
        else:
            plans[r["sample_id"]] = p
    if args.raw:  # decode the source-video frames first, grouped by video; extract() below then finds them on disk
        by_video = collections.defaultdict(list)
        for sid, p in plans.items():
            for n, idx, t in p[2]:
                f = os.path.join(D, "future_frames", f"{sid}_{t}.jpg")
                if n.startswith("RAW:") and not os.path.exists(f):
                    by_video[n[4:]].append((idx, f))
        print(f"source videos to decode: {len(by_video)}, frames: {sum(map(len, by_video.values()))}", flush=True)
        with Pool(args.workers) as pool:
            for k, (path, done, n) in enumerate(pool.imap_unordered(extract_raw, list(by_video.items())), 1):
                print(f"[raw {k}/{len(by_video)}] {os.path.basename(path)} {done}/{n}", flush=True)
    with Pool(args.workers) as pool:
        frames = dict(pool.imap_unordered(extract, [(sid, p[2]) for sid, p in plans.items()], chunksize=16))

    out = open(args.out, "w")
    nf = collections.Counter()
    for r in rows:
        sid = r["sample_id"]
        if sid not in plans or not frames.get(sid):
            continue
        rng = random.Random(1000003 * (sid + 1))
        opts = rng.sample(r["negative_answers"], 3) + [r["answer"]]
        rng.shuffle(opts)
        letter = "ABCD"[opts.index(r["answer"])]
        problem = "\n" + r["question"] + "\n" + "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts)) + "\n" + POST
        vb = r.get("video_basename")
        row = {
            "sample_id": sid, "video_id": r["video_id"], "task_goal": r["task_goal"],
            "problem": problem, "answer": letter,
            "videos": [os.path.join(D, "videos", "EpicKitchens", vb)] if vb else [],
            "images": [os.path.join(D, "images", "EpicKitchens", r["current_observation_basename"])],
            "opsd_future_frames": frames[sid], "opsd_future_text": plans[sid][1],
            "opsd_answer_text": f"{letter}. {r['answer']}",
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        nf[len(frames[sid])] += 1
        stats["written"] += 1
    out.close()
    print(dict(stats), "future frames per row:", dict(nf), flush=True)

    if args.no_sft:
        return
    # SFT on the official 6k chain-of-thought subset, in the OPSD / evaluation format
    sft = []
    for l in open(os.path.join(D, "annotations", "training_with_cot_6k.jsonl")):
        c = json.loads(l)
        content = c["prompt"][0]["content"]
        vids = [os.path.join(D, x["video"]) for x in content if x["type"] == "video"]
        ims = [os.path.join(D, x["image"]) for x in content if x["type"] == "image"]
        think = re.search(r"<think>(.*?)</think>", c["response"], re.S)
        letter = re.search(r"<answer>\s*([ABCD])", c["solution"])
        if not think or not letter:
            stats["sft skipped"] += 1
            continue
        user = "<video>" * len(vids) + "<image>" * len(ims) + "\n" + c["problem"].rstrip() + "\n" + POST
        sft.append({
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": f"<reason>\n{think.group(1).strip()}\n</reason>\n<answer>{letter.group(1)}</answer>"}],
            "videos": vids, "images": ims,
        })
    json.dump(sft, open(os.path.join(D, "egoplan_sft_cot6k.json"), "w"), ensure_ascii=False)
    print("sft rows:", len(sft), "skipped:", stats["sft skipped"])


if __name__ == "__main__":
    main()
