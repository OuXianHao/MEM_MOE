from __future__ import annotations

import re
from typing import List, Tuple


# -----------------------------
# 1) Action-layer prompt
#    Only decide NEXT ACTION:
#    - <search>...</search>
#    - <finish/>
# -----------------------------
SYSTEM_RULES = """You are a LOCAL multi-hop QA agent.
Use ONLY the provided local evidence in <history>. No outside knowledge.

At each step, output EXACTLY ONE action tag:
- <search>KEYWORDS</search>
- <finish/>

Rules:
1) If <history> does NOT contain enough facts yet, output <search>...</search>.
2) If <history> already contains enough facts to answer the question, output <finish/>.
3) Do NOT answer the question directly in this step.
4) Do NOT write "Answer:", "Search:", explanations, analysis, or any text outside the tag.
5) The FIRST character of your output must be '<'.
6) Do NOT repeat a previous <search> exactly.
7) Search query must be short keyword style only:
   - <= 8 words
   - <= 80 chars
   - no ':'
   - no quotes
   - no bullets
   - no full copied sentence from <history>
"""


# -----------------------------
# 2) Final-answer-layer prompt
#    Only produce final answer:
#    - <answer>...</answer>
# -----------------------------
FINAL_ANSWER_RULES = """You are a LOCAL QA answerer.
Use ONLY the evidence provided in <history>. No outside knowledge.

Output EXACTLY ONE tag:
<answer>FINAL ANSWER</answer>

Rules:
1) Answer the question directly and concisely.
2) Use ONLY facts explicitly supported by <history>.
3) Do NOT output "Answer:".
4) Do NOT output explanations before the tag.
5) The FIRST character of your output must be '<'.
6) If the evidence is still insufficient, output:
   <answer>unknown</answer>
"""


def make_step0_query(question: str) -> str:
    # Deterministic, entity-focused heuristic for the first forced search.
    stopwords = {
        "What", "Who", "Where", "When", "Why", "How", "Which",
        "Is", "Are", "Do", "Does", "Did", "Was", "Were", "Can", "Could"
    }

    caps = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*\b", question)
    filtered_caps = [w for w in caps if w not in stopwords]

    if filtered_caps:
        return " ".join(filtered_caps[:6])

    words = re.findall(r"[A-Za-z0-9]+", question.lower())
    return " ".join(words[:8])


def _truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ..."


def _format_prev_queries(history: List[Tuple[str, str]], keep_last: int = 8) -> str:
    prev_queries: List[str] = []
    for q, _ in history:
        q = (q or "").strip()
        if q and (not prev_queries or q != prev_queries[-1]):
            prev_queries.append(q)
    prev_queries = prev_queries[-keep_last:]
    return "\n".join(prev_queries) if prev_queries else "(none)"


def _format_history_blocks(
    history: List[Tuple[str, str]],
    keep_last_blocks: int = 4,
    max_chars_per_block: int = 1800,
) -> str:
    history = history[-keep_last_blocks:]

    blocks: List[str] = []
    for query, info_block in history:
        q = (query or "").strip()
        info = _truncate_text(info_block, max_chars=max_chars_per_block)
        blocks.append(f"<search>{q}</search>\n{info}")

    return "\n".join(blocks).strip() if blocks else "(empty)"


def build_state_prompt(question: str, history: List[Tuple[str, str]]) -> str:
    """
    Action-layer prompt.
    Only decide whether to SEARCH more or FINISH.
    Never generate the final answer here.
    """
    prev_text = _format_prev_queries(history, keep_last=8)
    history_text = _format_history_blocks(
        history,
        keep_last_blocks=4,
        max_chars_per_block=1800,
    )

    return (
        f"{SYSTEM_RULES}\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<prev_queries>\n{prev_text}\n</prev_queries>\n\n"
        f"<history>\n{history_text}\n</history>\n\n"
        "Output EXACTLY ONE action tag:\n"
        "- <search>KEYWORDS</search>\n"
        "- <finish/>\n\n"
        "Your output:\n"
    )


def build_final_answer_prompt(question: str, history: List[Tuple[str, str]]) -> str:
    """
    Final-answer-layer prompt.
    Use accumulated history/evidence to generate the final answer only.
    """
    history_text = _format_history_blocks(
        history,
        keep_last_blocks=6,
        max_chars_per_block=2200,
    )

    return (
        f"{FINAL_ANSWER_RULES}\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<history>\n{history_text}\n</history>\n\n"
        "Output EXACTLY ONE tag:\n"
        "<answer>FINAL ANSWER</answer>\n\n"
        "Your output:\n"
    )


def build_compression_prompt(question: str, information_block: str) -> str:
    """
    Memory/compression-layer prompt.
    Only extract trainable factual snippets from current evidence.
    """
    information_block = _truncate_text(information_block, max_chars=2400)

    return (
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<evidence>\n"
        f"{information_block}\n"
        "</evidence>\n\n"
        "Extract ONLY the key facts from <evidence> that help answer <question>.\n"
        "Write them inside EXACTLY ONE tag:\n"
        "<snippet>...</snippet>\n\n"
        "Rules:\n"
        "1) Use ONLY facts explicitly stated in <evidence>.\n"
        "2) 1 to 6 short lines, each <= 18 words.\n"
        "3) Each line must be a standalone fact (entity + relation + value).\n"
        "4) No analysis.\n"
        "5) No reasoning.\n"
        "6) No instructions.\n"
        "7) No citations like [1].\n"
        "8) No bullets like '-', '*', '1)'.\n"
        "9) If nothing useful exists, output exactly:\n"
        "<snippet>NONE</snippet>\n\n"
        "The FIRST character of your output must be '<'.\n"
        "Do not write any text before <snippet>.\n\n"
        "Your output:\n"
    )