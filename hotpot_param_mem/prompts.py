from __future__ import annotations

import re
from typing import List, Tuple


# 更像“师兄模板”的短规则：保留原语义，但压缩表达，减少模型复读/走神
SYSTEM_RULES = """You are a LOCAL multi-hop QA agent. Use ONLY the provided local evidence in <history>. No outside knowledge.

At each step, output EXACTLY ONE tag:
- <search>KEYWORDS</search>
- <answer>ANSWER</answer>

Rules:
1) Answer-first: If <history> contains enough facts to answer the question, output <answer>...</answer> immediately.
2) Evidence-only: Base your answer STRICTLY on facts in <history>. Do not hallucinate.
3) Missing-slot: If you cannot answer yet, output <search>KEYWORDS</search> to find the missing fact.
4) Anti-repeat: Do NOT repeat a previous <search> exactly.
5) Query format: short keyword query only (<=8 words, <=80 chars). No ':', no quotes, no bullets, no copying lines from <history>.
"""


def make_step0_query(question: str) -> str:
    # 过滤掉常见的疑问词，防止提取出 "What", "Who" 作为盲搜关键字
    stopwords = {"What", "Who", "Where", "When", "Why", "How", "Which", "Is", "Are", "Do", "Does", "Did"}
    
    # Deterministic, entity-focused heuristic.
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


def build_state_prompt(question: str, history: List[Tuple[str, str]]):
    # ---------- Previous queries (keep last 8, dedup consecutive) ----------
    prev_queries: List[str] = []
    for q, _ in history:
        q = (q or "").strip()
        if q and (not prev_queries or q != prev_queries[-1]):
            prev_queries.append(q)
    prev_queries = prev_queries[-8:]

    prev_text = "\n".join(prev_queries) if prev_queries else "(none)"

    # ---------- HISTORY: keep recent blocks to reduce noise (align with brother style) ----------
    # 只保留最后 4 个检索块（可按需调，但这是最稳的默认）
    history = history[-4:]

    blocks: List[str] = []
    for query, info_block in history:
        # info_block 可能很长；截断减少噪声（不改变逻辑，只减少干扰）
        info_block = _truncate_text(info_block, max_chars=1800)
        blocks.append(f"<search>{(query or '').strip()}</search>\n{info_block}")
    history_text = "\n".join(blocks).strip() if blocks else "(empty)"

    # ---------- Compose prompt (short, template-like) ----------
    return (
        f"{SYSTEM_RULES}\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<prev_queries>\n{prev_text}\n</prev_queries>\n\n"
        f"<history>\n{history_text}\n</history>\n\n"
        "Output ONLY ONE tag: <search>...</search> OR <answer>...</answer>.\n"
        "Your Output:\n"  # 👈 强引导词，逼迫模型开口
    )


def build_compression_prompt(question: str, information_block: str) -> str:
    return (
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<evidence>\n"
        f"{information_block}\n"
        "</evidence>\n\n"
        "Content for test-time training (TTT): extract ONLY the key facts from <evidence> that help answer <question>.\n"
        "Write the facts inside <snippet>...</snippet>.\n"
        "Rules:\n"
        "- Use ONLY facts explicitly stated in <evidence>.\n"
        "- 1 to 6 short lines, each <= 18 words.\n"
        "- Each line is a standalone fact (entity + relation + value).\n"
        "- No analysis, no reasoning, no instructions.\n"
        "- No citations like [1]. No bullets like '-', '*', '1)'.\n"
        "- If nothing useful, output exactly <snippet>NONE</snippet>.\n\n"
        "Your Output:\n" # 👈 不要在这里写 <snippet>，让模型自己生成！
    )