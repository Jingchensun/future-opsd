"""FutureOmni (OpenMOSS, arXiv 2601.13836): 1,034 multiple-choice questions on videos cut at the split point.
Prompt = the official one for the futureomni split (eval/infer_vllm.py, PROMPT_WITH_SIX_OPTION + "Question: ...
Answer: "); video only (Qwen3-VL has no audio input), 32 frames. Scoring: first option letter in the output."""
import ast
import os
import re
from collections import defaultdict

ROOT = os.path.expandvars(os.environ.get("FUTUREOMNI_ROOT", "${DATA_ROOT}/extra/futureomni"))
VIDEO_DIR = os.path.join(ROOT, "videos")

PROMPT = (
    "\n    These are the frames of a video and the corresponding audio.\n"
    "    Please answer the following multiple-choice question based on the video and audio content.\n"
    "    Choose the correct option and respond with ONLY the letter (A, B, C, D, E, F) of your choice.\n    "
)


def _options(doc):
    o = doc["options"]
    return ast.literal_eval(o) if isinstance(o, str) else list(o)


def futureomni_doc_to_visual(doc):
    return [os.path.join(VIDEO_DIR, f"{doc['qid']}.mp4")]


def futureomni_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_str = doc["question"] + "\n".join(_options(doc))  # sic: no newline before the options, as upstream
    return PROMPT + "Question: {}\nAnswer: ".format(question_str)


def _parse(pred, n):
    letters = "ABCDEF"[:n]
    s = pred.strip()
    m = re.match(rf"^\(?([{letters}])\b", s)
    if m:
        return m.group(1)
    m = re.search(rf"(?:answer is|answer:)\s*\(?([{letters}])\b", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{letters}])\b", s)
    return m.group(1) if m else ""


def futureomni_process_results(doc, results):
    pred = _parse(results[0], len(_options(doc)))
    rec = {"qid": doc["qid"], "pred": pred, "answer": doc["answer"], "audio_type": doc.get("audio_type"),
           "video_domain": doc.get("video_domain"), "forecasting_pattern": doc.get("forecasting_pattern")}
    return {"futureomni_acc": rec}


def futureomni_aggregate(results):
    by = defaultdict(list)
    for r in results:
        ok = r["pred"] == r["answer"]
        by["all"].append(ok)
        by[f"audio:{r['audio_type']}"].append(ok)
    from loguru import logger

    logger.info("FutureOmni " + " | ".join(f"{k} {100 * sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(by.items())))
    logger.info(f"FutureOmni unparsed predictions: {sum(1 for r in results if not r['pred'])}/{len(results)}")
    return 100 * sum(by["all"]) / len(by["all"])
