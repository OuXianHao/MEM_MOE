from __future__ import annotations

import re
from dataclasses import dataclass


# =========================
# Action-layer parsing
# Supports:
#   <search>...</search>
#   <search>...
#   <finish/>
#   <finish />
# Weak fallback:
#   Search: ...
#   Finish
# =========================

ACTION_SEARCH_CLOSED_RE = re.compile(
    r"<search>(.*?)</search>",
    re.DOTALL | re.IGNORECASE,
)

ACTION_SEARCH_OPEN_RE = re.compile(
    r"<search>(.*)$",
    re.DOTALL | re.IGNORECASE,
)

ACTION_FINISH_RE = re.compile(
    r"<finish\s*/>",
    re.IGNORECASE,
)

ACTION_SEARCH_WEAK_RE = re.compile(
    r"^\s*search\s*:\s*(.+?)\s*$",
    re.DOTALL | re.IGNORECASE,
)

ACTION_FINISH_WEAK_RE = re.compile(
    r"^\s*finish\s*$",
    re.DOTALL | re.IGNORECASE,
)


# =========================
# Final-answer-layer parsing
# Supports:
#   <answer>...</answer>
#   <answer>...
# Weak fallback:
#   Answer: ...
# =========================

ANSWER_CLOSED_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)

ANSWER_OPEN_RE = re.compile(
    r"<answer>(.*)$",
    re.DOTALL | re.IGNORECASE,
)

ANSWER_WEAK_RE = re.compile(
    r"^\s*answer\s*:\s*(.+?)\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedAction:
    action_type: str
    content: str
    forced_terminate: bool = False


def parse_first_action(text: str) -> ParsedAction:
    """
    Parse the NEXT ACTION only.

    Expected action protocol:
      - <search>...</search>
      - <finish/>

    Weak compatibility:
      - Search: ...
      - Finish
    """
    t = (text or "").strip()

    # 1) <finish/>
    m_finish = ACTION_FINISH_RE.search(t)
    if m_finish:
        return ParsedAction("finish", "", forced_terminate=False)

    # 2) <search>...</search>
    m_search = ACTION_SEARCH_CLOSED_RE.search(t)
    if m_search:
        content = (m_search.group(1) or "").strip()
        return ParsedAction("search", content, forced_terminate=False)

    # 3) <search>...   (stop-truncated)
    m_search_open = ACTION_SEARCH_OPEN_RE.search(t)
    if m_search_open:
        content = (m_search_open.group(1) or "").strip()
        return ParsedAction("search", content, forced_terminate=False)

    # 4) weak fallback: Search: ...
    m_search_weak = ACTION_SEARCH_WEAK_RE.search(t)
    if m_search_weak:
        content = (m_search_weak.group(1) or "").strip()
        return ParsedAction("search", content, forced_terminate=False)

    # 5) weak fallback: Finish
    m_finish_weak = ACTION_FINISH_WEAK_RE.search(t)
    if m_finish_weak:
        return ParsedAction("finish", "", forced_terminate=False)

    # 6) complete failure
    return ParsedAction("finish", "", forced_terminate=True)


def parse_final_answer(text: str) -> ParsedAction:
    """
    Parse the FINAL ANSWER only.

    Expected answer protocol:
      - <answer>...</answer>

    Weak compatibility:
      - Answer: ...
    """
    t = (text or "").strip()

    # 1) <answer>...</answer>
    m = ANSWER_CLOSED_RE.search(t)
    if m:
        content = (m.group(1) or "").strip()
        if not content:
            content = "unknown"
        return ParsedAction("answer", content, forced_terminate=False)

    # 2) <answer>...   (stop-truncated)
    m2 = ANSWER_OPEN_RE.search(t)
    if m2:
        content = (m2.group(1) or "").strip()
        if not content:
            content = "unknown"
        return ParsedAction("answer", content, forced_terminate=False)

    # 3) weak fallback: Answer: ...
    m3 = ANSWER_WEAK_RE.search(t)
    if m3:
        content = (m3.group(1) or "").strip()
        if not content:
            content = "unknown"
        return ParsedAction("answer", content, forced_terminate=False)

    # 4) complete failure
    return ParsedAction("answer", "unknown", forced_terminate=True)