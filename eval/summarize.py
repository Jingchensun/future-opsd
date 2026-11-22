"""Collect FutureBench / TwiFF-Bench numbers for one output/<TAG> dir into summary.json.

FutureBench columns follow the paper: 1-Hop / 2-Hop / 3-Hop / Interp. (= hop5) / AVG (= future_acc).
TwiFF-Bench: CoT (reasoning_mean) / Answer (answer_mean) / Avg. (overall_mean) from post_score.py.

Usage: python eval/summarize.py output/<TAG> [output/<TAG2> ...]
"""
import glob
import json
import os
import sys



def summarize(out_dir):
    tag = os.path.basename(os.path.normpath(out_dir))
    summary = {"tag": tag}
    # lmms-eval only reports future_acc in results.json, so per-hop accuracy is recomputed from the samples
    for samples in sorted(glob.glob(os.path.join(out_dir, "futurebench*", "**", "*samples_futurebench*.jsonl"), recursive=True)):
        task = os.path.basename(samples).split("samples_", 1)[1].rsplit(".jsonl", 1)[0]
        hits, tot = {}, {}
        for line in open(samples):
            r = json.loads(line)["future_acc"]
            for k in (r["question_type"], "all"):
                tot[k] = tot.get(k, 0) + 1
                hits[k] = hits.get(k, 0) + (r["pred_answer"] == r["answer"])
        keys = [("hop1", "1-Hop"), ("hop2", "2-Hop"), ("hop3", "3-Hop"), ("hop5", "Interp."), ("all", "AVG")]
        summary[task] = {name: round(100 * hits[k] / tot[k], 1) for k, name in keys if tot.get(k)}
        summary[task]["n"] = tot["all"]
    scored = sorted(glob.glob(os.path.join(out_dir, "twiffbench", "**", "*.summary.json"), recursive=True))
    if scored:
        s = json.load(open(scored[-1]))
        summary["twiffbench"] = {"CoT": s.get("reasoning_mean"), "Answer": s.get("answer_mean"), "Avg.": s.get("overall_mean"),
                                 "n_scored": s.get("n_scored_ok"), "n_err": s.get("n_scored_err")}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d), ensure_ascii=False))
