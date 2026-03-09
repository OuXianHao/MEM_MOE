from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List


class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.path, "a", encoding="utf-8")

    def write(self, record: Dict):
        self.fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self):
        self.fp.flush()

    def close(self):
        self.fp.close()


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Likely a truncated line due to crash/interruption; skip it.
                # You can uncomment the next line if you want visibility:
                # print(f"[WARN] Skipping invalid JSONL line {path}:{lineno}")
                continue
    return out

def write_summary(path: str, summary: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def summarize(eval_records: Iterable[Dict], extra: Dict) -> Dict:
    records = list(eval_records)
    n = len(records)
    em = sum(r.get("em", 0.0) for r in records) / n if n else 0.0
    f1 = sum(r.get("f1", 0.0) for r in records) / n if n else 0.0
    return {
        "completed_count": n,
        "em": em,
        "f1": f1,
        "updated_at": time.time(),
        **extra,
    }
