"""TTRL (Zuo et al., 2025) reward for 4-way multiple choice (EgoPlan-IT): label-free, majority vote as the pseudo-label.

The rollouts of one item (same uid) vote with their parsed option letters (egoplan_mcq_reward.extract_letter; empty
letters do not vote; ties go to the letter that reached the top count first). Each rollout then gets the E18b reward
against that pseudo-label instead of the dataset answer:
overall = (1 - format_weight) * (letter == majority) + format_weight * format, and -1 for degenerate repetition.
The dataset answer is used only for logging: accuracy (rollout vs answer), label_acc (majority vs answer).
"""
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REWARD_NAME = "egoplan_ttrl"
REWARD_TYPE = "batch"

_spec = importlib.util.spec_from_file_location("egoplan_mcq_reward", Path(__file__).with_name("egoplan_mcq_reward.py"))
_mcq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mcq)


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1, **_: Any) -> list[dict[str, float]]:
    letters = [_mcq.extract_letter(x["response"]) for x in reward_inputs]
    groups = defaultdict(list)
    for i, x in enumerate(reward_inputs):
        groups[x.get("uid", i)].append(i)
    majority, share = {}, {}
    for uid, idx in groups.items():
        votes = Counter(letters[i] for i in idx if letters[i])
        top = votes.most_common(1)
        majority[uid] = top[0][0] if top else ""
        share[uid] = top[0][1] / len(idx) if top else 0.0
    scores = []
    for i, x in enumerate(reward_inputs):
        uid = x.get("uid", i)
        response = x["response"]
        gt = str(x["ground_truth"]).strip().upper()[:1]
        fmt = 1.0 if _mcq.FORMAT_RE.match(response) else 0.0
        agree = 1.0 if letters[i] and letters[i] == majority[uid] else 0.0
        rep = _mcq._fl1.compute_repetition_penalty(response)
        scores.append({
            "overall": (1.0 - format_weight) * (-1.0 if rep > 0.5 else agree) + format_weight * fmt,
            "accuracy": 1.0 if letters[i] == gt else 0.0,
            "label_acc": 1.0 if majority[uid] == gt else 0.0,
            "majority_share": share[uid],
            "agree": agree,
            "format": fmt,
            "is_repetitive": float(rep > 0.5),
            "response_words": float(len(response.split())),
        })
    return scores
