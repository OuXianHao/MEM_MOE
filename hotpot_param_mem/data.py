from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compute_episode_id(example: Dict) -> str:
    if "_id" in example and example["_id"]:
        return str(example["_id"])
    if "id" in example and example["id"]:
        return str(example["id"])
    question = str(example.get("question", ""))
    context = json.dumps(example.get("context", []), ensure_ascii=False)
    return _stable_hash(f"{question}||{context}")


def load_hotpot_examples(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples: List[Dict] = []
    for raw in data:
        ex = dict(raw)
        ex["episode_id"] = compute_episode_id(ex)
        examples.append(ex)
    return examples


def chunk_for_worker(items: Sequence[Dict], worker_idx: int, total_workers: int) -> List[Dict]:
    return [item for i, item in enumerate(items) if i % total_workers == worker_idx]


def dedupe_and_sort_by_episode(records: Iterable[Dict]) -> List[Dict]:
    by_id: Dict[str, Dict] = {}
    for rec in records:
        episode_id = str(rec["episode_id"])
        by_id[episode_id] = rec
    return [by_id[k] for k in sorted(by_id.keys())]

def sort_trace_records(records: Iterable[Dict]) -> List[Dict]:
    """Sort step-level trace records stably by (episode_id, step_id). No dedupe."""
    def _key(rec: Dict):
        eid = str(rec.get("episode_id", ""))
        sid = rec.get("step_id", -1)
        try:
            sid = int(sid)
        except Exception:
            sid = -1
        return (eid, sid)
    return sorted(list(records), key=_key)