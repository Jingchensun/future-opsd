"""Logging-only reward for OPSD runs: the learning signal is the teacher's per-token log-prob, not this score.

Reports the answer format and response length so training can be monitored; overall is always 0.
"""
import re
from typing import Any

REWARD_NAME = "opsd_logging"
REWARD_TYPE = "sequential"

FORMAT_RE = re.compile(r"^\s*<reason>.*?</reason>\s*<answer>.*?</answer>\s*$", re.DOTALL)


def compute_score(reward_input: dict[str, Any], **_: Any) -> dict[str, float]:
    response = reward_input["response"]
    return {
        "overall": 0.0,
        "format": 1.0 if FORMAT_RE.match(response) else 0.0,
        "has_answer": 1.0 if "<answer>" in response and "</answer>" in response else 0.0,
        "response_words": float(len(response.split())),
    }
