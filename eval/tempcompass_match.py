"""TempCompass rule-based answer matchers (no GPT judge), used by eval/score_video3.py.

Copied verbatim from the matcher part of our earlier TempCompass rescorer (the scorer the
paper's TempCompass numbers were computed with). The task's own matcher only accepts exact forms ("Sentence A", "A", "A.");
these also understand an option index anywhere in the reply, a bare letter / number followed by ':' or ')', and a reply
that quotes (a prefix of) one option sentence. Replies that match nothing count as wrong.
"""
import glob, json, os, re, sys
LETTERS = "ABCD"

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def options_of(question):
    """-> list of (index_label, sentence) e.g. [("A","The man ..."),("B",...)] or [("1",...),("2",...)]"""
    out = []
    for line in question.split("\n")[1:]:
        m = re.match(r"^\s*(?:\(?)(?:Sentence|Caption|Option)?\s*([A-D1-4])[\)\.:]\s*(.*)$", line.strip())
        if m: out.append((m.group(1), m.group(2).strip()))
    return out

def match_index(text, opts):
    labels = [l for l, _ in opts]
    m = re.search(r"\b(?:Sentence|Caption|Option)\s*([A-D1-4])\b", text)
    if m and m.group(1) in labels: return m.group(1)
    m = re.match(r"^\s*\(?([A-D1-4])[\)\.:\s]", text)
    if m and m.group(1) in labels: return m.group(1)
    m = re.match(r"^\s*\(?([A-D1-4])\s*$", text)
    if m and m.group(1) in labels: return m.group(1)
    return None

def match_sentence(text, opts):
    # candidates: the whole reply, and the reply with a leading "The best option is:" / "Answer:" style prefix removed
    # (only a prefix that ends with a colon or the words option/caption/sentence/answer -- a bare "is" inside the sentence must not be stripped)
    cands = [norm(text)]
    m = re.match(r"^\s*(?:the\s+)?(?:best|correct|most\s+suitable|more\s+suitable)?\s*(?:answer|option|caption|sentence|description|match)?\s*(?:is|would\s+be)?\s*:?\s*(.*)$", text, flags=re.I | re.S)
    if m and m.group(1) and norm(m.group(1)) != cands[0]: cands.append(norm(m.group(1)))
    m2 = re.match(r"^[^:\u0027\"\u201c\u2018]{0,80}[:\u0027\"\u201c\u2018]\s*(.*)$", text, flags=re.S)
    if m2 and m2.group(1) and norm(m2.group(1)) and norm(m2.group(1)) not in cands: cands.append(norm(m2.group(1)))
    best, best_len = None, 0
    for t in cands:
        if not t: continue
        for label, sent in opts:
            s = norm(sent)
            if not s: continue
            # reply contains the option, or is a (truncated) prefix of it, or the option is a prefix of the reply
            if s in t or (len(t) >= 15 and s.startswith(t)) or (len(s) >= 15 and t.startswith(s)):
                if len(s) > best_len: best, best_len = label, len(s)
        if best: return best
    t = cands[-1] or cands[0]
    # word-overlap fallback: >= 70% of an option's words, and strictly more than the other option
    tw = set(t.split()); scores = []
    for label, sent in opts:
        sw = set(norm(sent).split()); scores.append((len(tw & sw) / max(1, len(sw)), label))
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 0.7 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.2): return scores[0][1]
    return None

def answer_label(answer, opts):
    m = re.match(r"^\s*(?:Sentence|Caption|Option)?\s*([A-D1-4])[\)\.:]", answer.strip())
    if m: return m.group(1)
    for label, sent in opts:
        if norm(sent) == norm(answer): return label
    return norm(answer)

def score_mc(doc, pred):            # multi-choice: answer like "B. normal speed"; options in question lines "A. ..."
    opts = options_of(doc["question"]); gt = answer_label(doc["answer"], opts)
    lab = match_index(pred, opts) or match_sentence(pred, opts)
    return (lab == gt) if lab else None

def score_cm(doc, pred):            # caption matching: answer like "Sentence B: ..." / "Option 1: ..."
    opts = options_of(doc["question"]); gt = answer_label(doc["answer"], opts)
    lab = match_index(pred, opts) or match_sentence(pred, opts)
    return (lab == gt) if lab else None

def score_yn(doc, pred):
    p = pred.strip().lower(); m = re.search(r"\b(yes|no)\b", p)
    return (m.group(1) == doc["answer"].strip().lower()) if m else None
