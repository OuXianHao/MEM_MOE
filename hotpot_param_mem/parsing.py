from __future__ import annotations

import re
from dataclasses import dataclass

# 先匹配完整闭合标签
ACTION_CLOSED_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
# vLLM stop 常见返回：不包含 stop string，本质是“开标签 + 内容到结尾”
ACTION_OPEN_RE = re.compile(r"<(search|answer)>(.*)$", re.DOTALL | re.IGNORECASE)


@dataclass
class ParsedAction:
    action_type: str
    content: str
    forced_terminate: bool = False


def parse_first_action(text: str) -> ParsedAction:
    t = (text or "").strip()

    # 1) 优先：闭合标签形式
    m = ACTION_CLOSED_RE.search(t)
    if m:
        action_type = m.group(1).lower()
        content = (m.group(2) or "").strip()
        if not content:
            content = "unknown" if action_type == "answer" else ""
        return ParsedAction(action_type, content, forced_terminate=False)

    # 2) 兼容：开标签到结尾（vLLM stop 截断导致没有 </search> / </answer>）
    m2 = ACTION_OPEN_RE.search(t)
    if m2:
        action_type = m2.group(1).lower()
        content = (m2.group(2) or "").strip()
        if not content:
            content = "unknown" if action_type == "answer" else ""
        # 注意：这不是失败，不要 forced_terminate
        return ParsedAction(action_type, content, forced_terminate=False)

    # 3) 完全找不到 action：按协议强制 unknown
    return ParsedAction("answer", "unknown", forced_terminate=True)