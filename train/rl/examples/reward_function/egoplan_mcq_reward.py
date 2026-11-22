"""Rule-based reward for 4-way multiple choice (EgoPlan-IT / SEED-Bench-R1 training): no judge model.

overall = (1 - format_weight) * correct + format_weight * format, and correct = -1 for degenerate repetition
(as cot_thinking_reward_function). The letter is parsed like the SEED-Bench-R1 evaluation
(lmms-eval seedbench_r1 extract_thinking_answer_regex): the first A-D inside <answer>...</answer>.
"""
import importlib.util
import re
from pathlib import Path
from typing import Any

REWARD_NAME = "egoplan_mcq"
REWARD_TYPE = "sequential"

_spec = importlib.util.spec_from_file_location("future_l1_reward_function", Path(__file__).with_name("future_l1_reward_function.py"))
_fl1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fl1)

FORMAT_RE = re.compile(r"^\s*<reason>.*?</reason>\s*<answer>.*?</answer>\s*$", re.DOTALL)


def extract_letter(response: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response.strip(), re.DOTALL)
    if not m:
        return ""
    ans = m.group(1).strip()
    if len(ans.split()) > 10 and not re.search("[ABCD]", ans):
        return ""
    m = re.search(r"[ABCD]", ans)
    return m.group(0) if m else ""


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.1, **_: Any) -> dict[str, float]:
    response = reward_input["response"]
    gt = str(reward_input["ground_truth"]).strip().upper()[:1]
    fmt = 1.0 if FORMAT_RE.match(response) else 0.0
    correct = 1.0 if extract_letter(response) == gt else 0.0
    rep = _fl1.compute_repetition_penalty(response)
    acc = -1.0 if rep > 0.5 else correct
    return {
        "overall": (1.0 - format_weight) * acc + format_weight * fmt,
        "accuracy": correct,
        "format": fmt,
        "is_repetitive": float(rep > 0.5),
        "response_words": float(len(response.split())),
    }
