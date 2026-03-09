from __future__ import annotations

import re
import string
from collections import Counter
from typing import Tuple


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(prediction: str, ground_truth: str) -> Tuple[float, float]:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    em = 1.0 if pred == gold else 0.0

    pred_toks = pred.split()
    gold_toks = gold.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        f1 = float(pred_toks == gold_toks)
    elif num_same == 0:
        f1 = 0.0
    else:
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
    return em, f1
