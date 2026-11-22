"""Score SEED-Bench-R1 / TempCompass / TemporalBench runs with the rules of our earlier evaluation code.

- SB-R1: task accuracy, mean over L1/L2/L3.
- TempCompass: robust rule-based matcher from our earlier TempCompass rescorer
  (its function definitions are executed from that file, so the rules stay identical); TC = mean of the 3 subtasks.
- TemporalBench: letter normalisation ("A." -> "A") as in rescore_temporalbench.py; binary accuracy and
  multiple-binary accuracy (a video counts only if all its questions are right). TB = short multi-binary.

Usage: python eval/score_video3.py output/<TAG> [output/<TAG2> ...]   -> writes <dir>/video3_summary.json
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tempcompass_match as _tc  # noqa: E402

TC_SCORERS = {"multi_choice": _tc.score_mc, "caption_matching": _tc.score_cm, "yes_no": _tc.score_yn}


def rows_of(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else None


def score(out_dir):
    s = {"tag": os.path.basename(os.path.normpath(out_dir))}
    lv = {}
    for level in ("l1", "l2", "l3"):
        f = f"{out_dir}/seedbench_r1/{level}/results.json"
        if os.path.exists(f):
            m = next(iter(json.load(open(f))["results"].values()))
            lv[level] = round(next(v for k, v in m.items() if k.startswith("seedbench_r1_acc")), 2)
    if lv:
        s["SB-R1"] = {**lv, "mean": round(sum(lv.values()) / len(lv), 2)}

    tc = {}
    for sub, fn in TC_SCORERS.items():
        rows = rows_of(f"{out_dir}/tempcompass/{sub}/samples.jsonl")
        if not rows:
            continue
        res = [fn(r["doc"], r["filtered_resps"][0]) for r in rows]
        tc[sub] = {"acc": round(100 * sum(1 for x in res if x) / len(rows), 2), "unmatched": sum(1 for x in res if x is None), "n": len(rows)}
    if tc:
        s["TC"] = {**tc, "mean": round(sum(v["acc"] for v in tc.values()) / len(tc), 2)}

    tb = {}
    for sub in ("short_qa", "long_qa"):
        rows = rows_of(f"{out_dir}/temporalbench/{sub}/samples.jsonl")
        if not rows:
            continue
        norm = lambda x: (x.strip().upper()[:1] if x.strip().upper()[:1] in ("A", "B") else x.strip().upper())
        ok = [r["doc"]["GT"].strip().upper() == norm(r["filtered_resps"][0]) for r in rows]
        per_video = collections.defaultdict(lambda: True)
        for r, f in zip(rows, ok):
            per_video[r["doc"]["video_name"]] &= f
        tb[sub] = {"binary": round(100 * sum(ok) / len(ok), 2), "multi_binary": round(100 * sum(per_video.values()) / len(per_video), 2), "n": len(rows)}
    f = f"{out_dir}/temporalbench/short_caption/results.json"
    if os.path.exists(f):
        tb["short_caption"] = next(iter(json.load(open(f))["results"].values()))
    if tb:
        s["TB"] = tb
    json.dump(s, open(f"{out_dir}/video3_summary.json", "w"), indent=1)
    return s


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(score(d), ensure_ascii=False))
