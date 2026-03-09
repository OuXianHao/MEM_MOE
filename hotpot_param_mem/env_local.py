from __future__ import annotations

import re
import string
from typing import List, Sequence, Tuple

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "with", "by", "at", "from", "that", "this", "which",
    "who", "what", "when", "where", "why", "how",
}

_punct_tbl = str.maketrans("", "", string.punctuation)



def normalize_tokens(text: str) -> List[str]:
    text = (text or "").lower().translate(_punct_tbl)
    text = re.sub(r"\s+", " ", text).strip()
    return [tok for tok in text.split(" ") if tok]


def build_paragraphs_from_context(context: Sequence) -> List[str]:
    paragraphs: List[str] = []
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sents = item[0], item[1]
            if isinstance(sents, list):
                para = f"{title}: " + " ".join(str(s) for s in sents)
            else:
                para = f"{title}: {sents}"
            para = para.strip()
            if para:
                paragraphs.append(para)
    return paragraphs


def retrieve_local(
    question: str,
    query: str,
    context: Sequence,
    topk: int,
    max_chars: int,
) -> Tuple[List[str], str]:
    paragraphs = build_paragraphs_from_context(context)

    # Use query tokens primarily; fall back to question if query is empty.
    q_tokens = set(normalize_tokens(query))
    if not q_tokens:
        q_tokens = set(normalize_tokens(question))

    scored: List[Tuple[int, int, str]] = []
    for idx, para in enumerate(paragraphs):
        ptoks = set(normalize_tokens(para))
        overlap = len(q_tokens.intersection(ptoks))
        scored.append((overlap, idx, para))

    # Sort by overlap desc, then by original index asc (stable & readable).
    scored.sort(key=lambda x: (-x[0], x[1]))

    selected: List[str] = []
    seen = set()
    for overlap, _, para in scored:
        # Optional noise control: only keep paragraphs with positive overlap.
        if overlap <= 0:
            break
        if para in seen:
            continue
        seen.add(para)
        selected.append(para[:max_chars])
        if len(selected) >= topk:
            break

    information = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(selected))
    return selected, f"<information>\n{information}\n</information>"


def keyword_overlap_ratio(question: str, snippet: str) -> float:
    q = set(normalize_tokens(question))
    if not q:
        return 0.0
    s = set(normalize_tokens(snippet))
    return len(q.intersection(s)) / max(1, len(q))